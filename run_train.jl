#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
# JuliaGPT Local Training Runner
# Extracts and runs all notebook cells with optimizations:
#   - Resource sensing (CPU/RAM/GPU) with configurable limits
#   - Unsloth-inspired sequence packing
#   - Pre-allocated KV cache, single @diff tape, typed params
#   - Periodic GC for memory stability
# Usage: julia --threads=auto run_train.jl
# ═══════════════════════════════════════════════════════════════

println("JuliaGPT Local Training Runner")
println("=" ^ 50)

# ═══════════════════════════════════════════════════════════════
# 0. Resource Sensing & Auto-Configuration
# ═══════════════════════════════════════════════════════════════

RESOURCE_LIMIT = 0.50  # Use at most 50% of available resources

using LinearAlgebra

total_threads = Sys.CPU_THREADS
total_cores = div(total_threads, 2)
max_blas_threads = max(1, Int(floor(total_cores * RESOURCE_LIMIT)))
BLAS.set_num_threads(max_blas_threads)

println("\n=== Resource Configuration ($(Int(RESOURCE_LIMIT * 100))% limit) ===")
println("CPU: $(total_cores) cores / $(total_threads) threads")
println("  BLAS threads: $max_blas_threads")

total_ram_gb = round(Sys.total_memory() / 1024^3, digits=1)
free_ram_gb = round(Sys.free_memory() / 1024^3, digits=1)
ram_limit_gb = round(free_ram_gb * RESOURCE_LIMIT, digits=1)
println("RAM: $(total_ram_gb) GB total, $(free_ram_gb) GB free, limit: $(ram_limit_gb) GB")

HAS_GPU = false
try
    using CUDA
    if CUDA.functional()
        HAS_GPU = true
        gpu_name = CUDA.name(CUDA.device())
        gpu_total = round(CUDA.total_memory() / 1024^3, digits=2)
        gpu_free = round(CUDA.available_memory() / 1024^3, digits=2)
        gpu_limit = round(gpu_free * RESOURCE_LIMIT, digits=2)
        CUDA.pool.max_memory!(Int(floor(gpu_limit * 1024^3)))
        println("GPU: $gpu_name ($gpu_total GB, limit: $gpu_limit GB)")
    end
catch; end
if !HAS_GPU
    println("GPU: not available (using CPU)")
end

GC_INTERVAL = 100
ENABLE_PACKING = true
println("Optimizations: packing=$(ENABLE_PACKING), GC every $(GC_INTERVAL) steps")
println("=" ^ 50)

# ═══════════════════════════════════════════════════════════════
# 1. Packages & AutoGrad Setup
# ═══════════════════════════════════════════════════════════════

using Random, Printf, JSON3, AutoGrad, Statistics
using Downloads

Random.seed!(42)
println("\nAutoGrad.jl loaded")

# ═══════════════════════════════════════════════════════════════
# 2. Primitives
# ═══════════════════════════════════════════════════════════════

relu_ag(x) = max.(x, Float32(0))

function rmsnorm_ag(x)
    n = length(x)
    ms = sum(x .* x) / n
    scale = (ms + Float32(1e-5)) ^ Float32(-0.5)
    return x .* scale
end

function softmax_ag(logits)
    mx = maximum(logits)
    exps = exp.(logits .- mx)
    s = sum(exps)
    return exps ./ s
end

function to_dense_grad(g)
    if g isa AutoGrad.Sparse
        Float32.(AutoGrad.full(g))
    else
        Float32.(g)
    end
end

# ═══════════════════════════════════════════════════════════════
# 3. Helpers
# ═══════════════════════════════════════════════════════════════

function get_param_keys(n_layer::Int)
    keys = ["wte", "wpe", "lm_head"]
    for i in 0:n_layer-1
        append!(keys, [
            "layer$i.attn_wq", "layer$i.attn_wk", "layer$i.attn_wv", "layer$i.attn_wo",
            "layer$i.mlp_fc1", "layer$i.mlp_fc2"
        ])
    end
    return keys
end

function init_param(nout::Int, nin::Int; std=0.08f0)
    Param(randn(Float32, nout, nin) .* std)
end

function collect_params(state_dict, param_keys)
    [state_dict[key] for key in param_keys]
end

# ═══════════════════════════════════════════════════════════════
# 4. GPT Forward Pass (optimized)
# ═══════════════════════════════════════════════════════════════

function gpt(token_id::Int, pos_id::Int,
             kv_key_mats::Vector{Matrix{Float32}},
             kv_val_mats::Vector{Matrix{Float32}},
             kv_lens::Vector{Int},
             params, n_layer::Int, n_head::Int, head_dim::Int)

    tok_emb = params.wte[token_id, :]
    pos_emb = params.wpe[pos_id, :]
    x = tok_emb .+ pos_emb
    x = rmsnorm_ag(x)

    for li in 1:n_layer
        layer = params.layers[li]
        x_res = x
        x = rmsnorm_ag(x)

        q = layer.attn_wq * x
        k = layer.attn_wk * x
        v = layer.attn_wv * x

        k_detached = Float32.(value(k))
        v_detached = Float32.(value(v))

        idx = kv_lens[li] + 1
        kv_key_mats[li][:, idx] = k_detached
        kv_val_mats[li][:, idx] = v_detached
        kv_lens[li] = idx

        K_mat = @view kv_key_mats[li][:, 1:idx]
        V_mat = @view kv_val_mats[li][:, 1:idx]

        head_results = ntuple(n_head) do hh
            h = hh - 1
            hs = h * head_dim + 1
            he = hs + head_dim - 1
            q_h = q[hs:he]
            K_h = K_mat[hs:he, :]
            V_h = V_mat[hs:he, :]
            scores = (K_h' * q_h) ./ Float32(sqrt(head_dim))
            attn_w = softmax_ag(scores)
            V_h * attn_w
        end

        x_attn = vcat(head_results...)
        x = layer.attn_wo * x_attn
        x = x .+ x_res

        x_res = x
        x = rmsnorm_ag(x)
        x = layer.mlp_fc1 * x
        x = relu_ag(x)
        x = layer.mlp_fc2 * x
        x = x .+ x_res
    end

    return params.lm_head * x
end

function build_params(state_dict, n_layer::Int)
    layers = ntuple(n_layer) do li
        i = li - 1
        (attn_wq=state_dict["layer$i.attn_wq"], attn_wk=state_dict["layer$i.attn_wk"],
         attn_wv=state_dict["layer$i.attn_wv"], attn_wo=state_dict["layer$i.attn_wo"],
         mlp_fc1=state_dict["layer$i.mlp_fc1"], mlp_fc2=state_dict["layer$i.mlp_fc2"])
    end
    (wte=state_dict["wte"], wpe=state_dict["wpe"], lm_head=state_dict["lm_head"], layers=layers)
end

function alloc_kv_cache(n_layer::Int, d_model::Int, block_size::Int)
    ([zeros(Float32, d_model, block_size) for _ in 1:n_layer],
     [zeros(Float32, d_model, block_size) for _ in 1:n_layer],
     zeros(Int, n_layer))
end

reset_kv_cache!(kv_lens::Vector{Int}) = (kv_lens .= 0)

# ═══════════════════════════════════════════════════════════════
# 5. Checkpoint Save/Load
# ═══════════════════════════════════════════════════════════════

function save_checkpoint(path::String, state_dict, param_keys, uchars, hyperparams;
                         adam_m=nothing, adam_v=nothing, step::Int=0,
                         lr::Float64=0.01, b1::Float64=0.85, b2::Float64=0.99,
                         best_val_loss::Float64=Inf,
                         train_losses::Vector{Float64}=Float64[],
                         val_losses::Vector{Float64}=Float64[],
                         total_steps::Int=0, num_steps_target::Int=0)
    sd_data = Dict{String,Any}()
    for k in param_keys
        W = value(state_dict[k])
        sd_data[k] = [Float64.(W[i, :]) for i in 1:size(W, 1)]
    end
    adam_m_data = Dict{String,Any}()
    adam_v_data = Dict{String,Any}()
    if adam_m !== nothing
        for k in param_keys
            haskey(adam_m, k) || continue
            adam_m_data[k] = [Float64.(adam_m[k][i, :]) for i in 1:size(adam_m[k], 1)]
            adam_v_data[k] = [Float64.(adam_v[k][i, :]) for i in 1:size(adam_v[k], 1)]
        end
    end
    checkpoint = Dict{String,Any}(
        "format" => "autograd_v2",
        "uchars" => [string(c) for c in uchars],
        "hyperparams" => hyperparams,
        "state_dict" => sd_data,
        "optimizer" => Dict{String,Any}(
            "adam_m" => adam_m_data, "adam_v" => adam_v_data,
            "step" => step, "lr" => lr, "beta1" => b1, "beta2" => b2),
        "training" => Dict{String,Any}(
            "best_val_loss" => best_val_loss,
            "train_losses" => train_losses, "val_losses" => val_losses,
            "total_steps_completed" => total_steps, "num_steps_target" => num_steps_target))
    # Replace Inf with a large number for JSON compatibility
    if best_val_loss == Inf
        checkpoint["training"]["best_val_loss"] = 1e30
    end
    mkpath(dirname(path))
    open(path, "w") do f; JSON3.write(f, checkpoint); end
    vl_str = best_val_loss == Inf ? "Inf" : @sprintf("%.4f", best_val_loss)
    println("Checkpoint saved: $path (step $step, best_val_loss=$vl_str)")
end

function load_checkpoint(path::String)
    println("Loading checkpoint from $path ...")
    raw = JSON3.read(read(path, String))
    uchars = [only(String(s)) for s in raw["uchars"]]
    BOS = length(uchars) + 1
    hp = raw["hyperparams"]
    n_layer = Int(hp["n_layer"]); n_embd = Int(hp["n_embd"])
    block_size = Int(hp["block_size"]); n_head = Int(hp["n_head"])
    head_dim = n_embd ÷ n_head
    fmt = haskey(raw, "format") ? String(raw["format"]) : "v1"
    state_dict = Dict{String, Any}()
    for (key, matrix_rows) in pairs(raw["state_dict"])
        rows = [Float32.(collect(row)) for row in matrix_rows]
        W = vcat([reshape(r, 1, :) for r in rows]...)
        state_dict[string(key)] = Param(W)
    end
    opt_raw = raw["optimizer"]
    adam_m = Dict{String, Matrix{Float32}}()
    adam_v = Dict{String, Matrix{Float32}}()
    if fmt == "autograd_v2" && haskey(opt_raw, "adam_m") && !isempty(opt_raw["adam_m"])
        for (key, mr) in pairs(opt_raw["adam_m"])
            rows = [Float32.(collect(row)) for row in mr]
            adam_m[string(key)] = vcat([reshape(r, 1, :) for r in rows]...)
        end
        for (key, mr) in pairs(opt_raw["adam_v"])
            rows = [Float32.(collect(row)) for row in mr]
            adam_v[string(key)] = vcat([reshape(r, 1, :) for r in rows]...)
        end
    end
    step = Int(opt_raw["step"]); lr = Float64(opt_raw["lr"])
    b1 = Float64(opt_raw["beta1"]); b2 = Float64(opt_raw["beta2"])
    trn = raw["training"]
    best_val_loss = Float64(trn["best_val_loss"])
    train_losses = Float64.(collect(trn["train_losses"]))
    val_losses = Float64.(collect(trn["val_losses"]))
    total_steps = Int(trn["total_steps_completed"])
    num_steps_target = Int(trn["num_steps_target"])
    println("  vocab=$(BOS), embd=$n_embd, layers=$n_layer, step=$step")
    return (; state_dict, uchars, BOS, vocab_size=BOS,
              n_layer, n_embd, block_size, n_head, head_dim,
              adam_m, adam_v, step, lr, b1, b2,
              best_val_loss, train_losses, val_losses, total_steps, num_steps_target)
end

# ═══════════════════════════════════════════════════════════════
# 6. Dataset Download & Load
# ═══════════════════════════════════════════════════════════════

function download_and_clean(url::String, fn::String; is_gutenberg=true)
    if !isfile(fn)
        println("Downloading $fn ...")
        try; Downloads.download(url, fn)
        catch e; @warn "Download failed: $url -> $e"; return ""; end
    end
    txt = read(fn, String)
    if is_gutenberg
        txt = replace(txt, r"(?is)^.*?\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*{3}[\r\n]*" => "")
        txt = replace(txt, r"(?is)\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG.*$" => "")
    end
    txt = replace(txt, r"\r\n" => "\n")
    txt = replace(txt, r"\n{3,}" => "\n\n")
    return strip(txt)
end

println("\n--- Downloading training data ---")

sources = Dict(
    "grammar"=>("https://www.gutenberg.org/files/15665/15665-0.txt","latin_grammar.txt",true),
    "categories"=>("https://www.gutenberg.org/ebooks/2412.txt.utf-8","aristotle_categories.txt",true),
    "rhetoric"=>("http://classics.mit.edu/Aristotle/rhetoric.mb.txt","aristotle_rhetoric.txt",false),
    "prior_analytics"=>("http://classics.mit.edu/Aristotle/prior.mb.txt","prior_analytics.txt",false),
    "posterior_analytics"=>("http://classics.mit.edu/Aristotle/posterior.mb.txt","posterior_analytics.txt",false),
    "topics"=>("http://classics.mit.edu/Aristotle/topics.mb.txt","topics.txt",false),
    "boethius"=>("https://www.gutenberg.org/files/14328/14328-0.txt","boethius_consolation.txt",true),
    "heavens"=>("http://classics.mit.edu/Aristotle/heavens.mb.txt","aristotle_heavens.txt",false),
    "republic"=>("https://www.gutenberg.org/files/1497/1497-0.txt","plato_republic.txt",true),
    "apology"=>("https://www.gutenberg.org/files/1656/1656-0.txt","plato_apology.txt",true),
    "ethics"=>("https://www.gutenberg.org/files/8438/8438-0.txt","aristotle_ethics.txt",true),
    "emerson"=>("https://www.gutenberg.org/files/2944/2944-0.txt","emerson_essays.txt",true),
    "walden"=>("https://www.gutenberg.org/files/205/205-0.txt","thoreau_walden.txt",true),
    "epicurus"=>("https://www.gutenberg.org/files/57342/57342-0.txt","diogenes_epicurus.txt",true),
    "plato_symposium"=>("https://www.gutenberg.org/ebooks/1600.txt.utf-8","plato_symposium.txt",true),
    "plato_phaedo"=>("https://www.gutenberg.org/ebooks/1658.txt.utf-8","plato_phaedo.txt",true),
    "plato_crito"=>("https://www.gutenberg.org/ebooks/1657.txt.utf-8","plato_crito.txt",true),
    "plato_meno"=>("https://www.gutenberg.org/ebooks/1643.txt.utf-8","plato_meno.txt",true),
    "plato_phaedrus"=>("https://www.gutenberg.org/ebooks/1636.txt.utf-8","plato_phaedrus.txt",true),
    "plato_timaeus"=>("https://www.gutenberg.org/ebooks/1572.txt.utf-8","plato_timaeus.txt",true),
    "plato_laws"=>("https://www.gutenberg.org/ebooks/1750.txt.utf-8","plato_laws.txt",true),
    "plato_gorgias"=>("https://www.gutenberg.org/ebooks/1672.txt.utf-8","plato_gorgias.txt",true),
    "plato_protagoras"=>("https://www.gutenberg.org/ebooks/1591.txt.utf-8","plato_protagoras.txt",true),
    "plato_theaetetus"=>("https://www.gutenberg.org/ebooks/1726.txt.utf-8","plato_theaetetus.txt",true),
    "aristotle_physics"=>("http://classics.mit.edu/Aristotle/physics.mb.txt","aristotle_physics.txt",false),
    "aristotle_metaphysics"=>("http://classics.mit.edu/Aristotle/metaphysics.mb.txt","aristotle_metaphysics.txt",false),
    "aristotle_soul"=>("http://classics.mit.edu/Aristotle/soul.mb.txt","aristotle_soul.txt",false),
    "aristotle_poetics"=>("https://www.gutenberg.org/files/1974/1974.txt","aristotle_poetics.txt",true),
    "aristotle_politics"=>("https://www.gutenberg.org/ebooks/6762.txt.utf-8","aristotle_politics.txt",true),
    "aristotle_generation"=>("http://classics.mit.edu/Aristotle/gener_corr.mb.txt","aristotle_generation.txt",false),
    "marcus_meditations"=>("https://www.gutenberg.org/files/2680/2680-0.txt","marcus_meditations.txt",true),
    "epictetus_discourses"=>("https://www.gutenberg.org/ebooks/10661.txt.utf-8","epictetus_discourses.txt",true),
    "epictetus_enchiridion"=>("https://www.gutenberg.org/ebooks/45109.txt.utf-8","epictetus_enchiridion.txt",true),
    "seneca_moral_essays"=>("https://www.gutenberg.org/ebooks/64576.txt.utf-8","seneca_moral_essays.txt",true),
    "lucretius"=>("https://www.gutenberg.org/ebooks/785.txt.utf-8","lucretius_nature.txt",true),
    "cicero_duties"=>("https://www.gutenberg.org/ebooks/47001.txt.utf-8","cicero_duties.txt",true),
    "cicero_nature_gods"=>("https://www.gutenberg.org/files/14988/14988.txt","cicero_nature_gods.txt",true),
    "cicero_friendship"=>("https://www.gutenberg.org/ebooks/2808.txt.utf-8","cicero_friendship.txt",true),
    "descartes_method"=>("https://www.gutenberg.org/ebooks/59.txt.utf-8","descartes_method.txt",true),
    "descartes_meditations"=>("https://www.gutenberg.org/ebooks/70091.txt.utf-8","descartes_meditations.txt",true),
    "kant_pure_reason"=>("https://www.gutenberg.org/ebooks/4280.txt.utf-8","kant_pure_reason.txt",true),
    "spinoza_ethics"=>("https://www.gutenberg.org/ebooks/3800.txt.utf-8","spinoza_ethics.txt",true),
    "hobbes_leviathan"=>("https://www.gutenberg.org/ebooks/3207.txt.utf-8","hobbes_leviathan.txt",true),
    "locke_government"=>("https://www.gutenberg.org/ebooks/7370.txt.utf-8","locke_government.txt",true),
    "hume_understanding"=>("https://www.gutenberg.org/ebooks/9662.txt.utf-8","hume_understanding.txt",true),
    "rousseau_social_contract"=>("https://www.gutenberg.org/files/46333/46333-0.txt","rousseau_social_contract.txt",true),
    "nietzsche_beyond"=>("https://www.gutenberg.org/ebooks/4363.txt.utf-8","nietzsche_beyond.txt",true),
    "nietzsche_zarathustra"=>("https://www.gutenberg.org/ebooks/1998.txt.utf-8","nietzsche_zarathustra.txt",true),
    "mill_liberty"=>("https://www.gutenberg.org/ebooks/34901.txt.utf-8","mill_liberty.txt",true),
    "mill_utilitarianism"=>("https://www.gutenberg.org/ebooks/11224.txt.utf-8","mill_utilitarianism.txt",true),
    "machiavelli_prince"=>("https://www.gutenberg.org/files/57037/57037-0.txt","machiavelli_prince.txt",true),
    "bacon_essays"=>("https://www.gutenberg.org/ebooks/575.txt.utf-8","bacon_essays.txt",true),
    "montaigne_essays"=>("https://www.gutenberg.org/ebooks/3600.txt.utf-8","montaigne_essays.txt",true),
    "schopenhauer_essays"=>("https://www.gutenberg.org/ebooks/11945.txt.utf-8","schopenhauer_essays.txt",true),
)

texts = Dict{String,String}()
for (key, (url, fn, is_gut)) in sources
    texts[key] = download_and_clean(url, fn; is_gutenberg=is_gut)
end
println("Downloaded $(length(texts)) texts.")

# Unicode normalization
for (_, (_, fn, _)) in sources
    isfile(fn) || continue
    txt = read(fn, String)
    txt = replace(txt, "\u201c"=>"\"", "\u201d"=>"\"", "\u2018"=>"'", "\u2019"=>"'",
                       "\u2014"=>"--", "\u2013"=>"-", "\u2026"=>"...", "\u00A0"=>" ")
    txt = replace(txt, r"\n{3,}" => "\n\n")
    open(fn, "w") do io; write(io, strip(txt)); end
end

# Build TRAINING_DATA
TRAINING_DATA = String[]
for (_, (_, fn, _)) in sources
    isfile(fn) || continue
    txt = lowercase(read(fn, String))
    txt = replace(txt, r"[^a-z \.\n]" => " ")
    txt = replace(txt, r"  +" => " ")
    for p in split(txt, r"\n\n+")
        cleaned = strip(replace(String(p), r"\n" => " "))
        cleaned = replace(cleaned, r"  +" => " ")
        length(cleaned) >= 20 || continue
        while length(cleaned) > 512
            cutoff = min(512, length(cleaned))
            dot_pos = findlast('.', cleaned[1:cutoff])
            if dot_pos !== nothing && dot_pos > 100
                push!(TRAINING_DATA, strip(cleaned[1:dot_pos]))
                cleaned = strip(cleaned[dot_pos+1:end])
            else
                push!(TRAINING_DATA, strip(cleaned[1:cutoff]))
                cleaned = strip(cleaned[cutoff+1:end])
            end
        end
        length(cleaned) >= 20 && push!(TRAINING_DATA, cleaned)
    end
end
println("TRAINING_DATA: $(length(TRAINING_DATA)) paragraphs, $(sum(length, TRAINING_DATA)) chars")

# ═══════════════════════════════════════════════════════════════
# 7. Setup — Tokenizer, Packing, Parameters
# ═══════════════════════════════════════════════════════════════

docs = copy(TRAINING_DATA)
split_idx = max(1, Int(floor(0.9 * length(docs))))
train_docs = docs[1:split_idx]
val_docs = docs[split_idx+1:end]
if isempty(val_docs)
    val_docs = docs[max(1, end-4):end]
    train_docs = docs[1:max(1, end-5)]
end
println("train: $(length(train_docs)) docs, val: $(length(val_docs)) docs")

uchars = sort(unique(join(docs)))
BOS = length(uchars) + 1
vocab_size = BOS
println("vocab size: $vocab_size ($(length(uchars)) chars + BOS)")

char_to_id = Dict{Char, Int}(ch => i for (i, ch) in enumerate(uchars))

function tokenize_doc(doc::String, char_to_id::Dict{Char,Int}, BOS::Int)
    vcat([BOS], [char_to_id[ch] for ch in doc], [BOS])
end

n_layer    = 1
n_embd     = 16
block_size = 256
n_head     = 4
head_dim   = n_embd ÷ n_head

# Pre-tokenize
train_tokens_raw = [tokenize_doc(doc, char_to_id, BOS) for doc in train_docs]
val_tokens = [tokenize_doc(doc, char_to_id, BOS) for doc in val_docs]

# ── Unsloth-inspired Sequence Packing ──
function pack_sequences(token_seqs::Vector{Vector{Int}}, block_size::Int, BOS::Int)
    packed = Vector{Vector{Int}}()
    buffer = Int[]
    for seq in token_seqs
        if length(buffer) + length(seq) > block_size + 1 && !isempty(buffer)
            push!(packed, buffer[1:min(block_size+1, length(buffer))])
            buffer = Int[]
        end
        if length(seq) > block_size + 1 && isempty(buffer)
            push!(packed, seq)
            continue
        end
        append!(buffer, seq)
    end
    length(buffer) > 2 && push!(packed, buffer[1:min(block_size+1, length(buffer))])
    return packed
end

if ENABLE_PACKING
    train_tokens = pack_sequences(train_tokens_raw, block_size, BOS)
    avg_len = round(mean(length.(train_tokens)), digits=1)
    println("Packing: $(length(train_tokens_raw)) docs -> $(length(train_tokens)) packed (avg len $avg_len)")
else
    train_tokens = train_tokens_raw
end

hyperparams = Dict{String,Any}("n_layer"=>n_layer, "n_embd"=>n_embd, "block_size"=>block_size, "n_head"=>n_head)

state_dict = Dict{String, Any}()
state_dict["wte"] = init_param(vocab_size, n_embd)
state_dict["wpe"] = init_param(block_size, n_embd)
state_dict["lm_head"] = init_param(vocab_size, n_embd)
for i in 0:n_layer-1
    state_dict["layer$i.attn_wq"] = init_param(n_embd, n_embd)
    state_dict["layer$i.attn_wk"] = init_param(n_embd, n_embd)
    state_dict["layer$i.attn_wv"] = init_param(n_embd, n_embd)
    state_dict["layer$i.attn_wo"] = init_param(n_embd, n_embd)
    state_dict["layer$i.mlp_fc1"] = init_param(4n_embd, n_embd)
    state_dict["layer$i.mlp_fc2"] = init_param(n_embd, 4n_embd)
end

param_keys = get_param_keys(n_layer)
total_num_params = sum(length(value(state_dict[k])) for k in param_keys)
println("num params: $total_num_params")

params = build_params(state_dict, n_layer)

# ═══════════════════════════════════════════════════════════════
# 8. Validation
# ═══════════════════════════════════════════════════════════════

function compute_val_loss(val_tokens, params, BOS, block_size, n_layer, n_head, head_dim, n_embd)
    total_loss = 0.0; total_tokens = 0
    kv_km, kv_vm, kv_l = alloc_kv_cache(n_layer, n_embd, block_size)
    for tokens in val_tokens
        n = min(block_size, length(tokens) - 1)
        reset_kv_cache!(kv_l)
        for pos in 1:n
            logits = gpt(tokens[pos], pos, kv_km, kv_vm, kv_l, params, n_layer, n_head, head_dim)
            probs = softmax_ag(logits)
            p_val = Float64.(value(probs))
            total_loss += -log(max(p_val[tokens[pos+1]], 1e-10))
            total_tokens += 1
        end
    end
    return total_loss / max(total_tokens, 1)
end

# ═══════════════════════════════════════════════════════════════
# 9. Training Loop
# ═══════════════════════════════════════════════════════════════

function train_loop!(state_dict, params, param_keys, train_tokens, val_tokens,
                     adam_m, adam_v, uchars, hyperparams;
                     num_steps::Int, lr::Float64, b1::Float64, b2::Float64, eps::Float64,
                     n_layer::Int, n_head::Int, head_dim::Int, n_embd::Int,
                     block_size::Int, BOS::Int, gc_interval::Int=100,
                     best_val_loss::Float64=Inf,
                     train_loss_history::Vector{Float64}=Float64[],
                     val_loss_history::Vector{Float64}=Float64[],
                     start_step::Int=1)

    kv_km, kv_vm, kv_l = alloc_kv_cache(n_layer, n_embd, block_size)
    end_step = start_step + num_steps - 1

    println("--- training $num_steps steps ($start_step..$end_step) ---")
    t_start = time()
    last_save_time = time()
    completed_steps = start_step - 1

    try
        for step in start_step:end_step
            completed_steps = step
            tokens = train_tokens[mod1(step, length(train_tokens))]
            n = min(block_size, length(tokens) - 1)

            reset_kv_cache!(kv_l)

            # Single @diff tape for entire sequence
            tape = @diff begin
                loss_sum = Float32(0)
                for pos in 1:n
                    logits = gpt(tokens[pos], pos, kv_km, kv_vm, kv_l, params, n_layer, n_head, head_dim)
                    probs = softmax_ag(logits)
                    loss_sum = loss_sum + (-log(probs[tokens[pos+1]]))
                end
                loss_sum / Float32(n)
            end

            avg_loss = Float64(value(tape))
            push!(train_loss_history, avg_loss)

            lr_t = lr * (1 - (step - 1) / end_step)
            for k in param_keys
                g = grad(tape, state_dict[k])
                g === nothing && continue
                g_dense = to_dense_grad(g)
                adam_m[k] .= Float32(b1) .* adam_m[k] .+ Float32(1-b1) .* g_dense
                adam_v[k] .= Float32(b2) .* adam_v[k] .+ Float32(1-b2) .* g_dense .^ 2
                m_hat = adam_m[k] ./ Float32(1-b1^step)
                v_hat = adam_v[k] ./ Float32(1-b2^step)
                value(state_dict[k]) .-= Float32(lr_t) .* m_hat ./ (sqrt.(v_hat) .+ Float32(eps))
            end

            # Periodic GC (Unsloth pattern: controlled memory cleanup)
            if step % gc_interval == 0
                GC.gc(false)
            end

            if step % 50 == 0
                val_loss = compute_val_loss(val_tokens, params, BOS, block_size, n_layer, n_head, head_dim, n_embd)
                push!(val_loss_history, val_loss)
                elapsed = time() - t_start
                improved = ""
                if val_loss < best_val_loss
                    best_val_loss = val_loss
                    save_checkpoint("checkpoints/best_model.json", state_dict, param_keys, uchars, hyperparams;
                        adam_m=adam_m, adam_v=adam_v, step=step, lr=lr, b1=b1, b2=b2,
                        best_val_loss=best_val_loss, train_losses=train_loss_history,
                        val_losses=val_loss_history, total_steps=step, num_steps_target=end_step)
                    improved = " << new best!"
                end
                @printf("step %5d / %5d | train %.4f | val %.4f | %.1fs%s\n",
                        step, end_step, avg_loss, val_loss, elapsed, improved)
            elseif step % 10 == 0
                @printf("step %5d / %5d | train %.4f | %.1fs\n", step, end_step, avg_loss, time()-t_start)
            end

            if step % 100 == 0
                save_checkpoint("checkpoints/checkpoint_latest.json", state_dict, param_keys, uchars, hyperparams;
                    adam_m=adam_m, adam_v=adam_v, step=step, lr=lr, b1=b1, b2=b2,
                    best_val_loss=best_val_loss, train_losses=train_loss_history,
                    val_losses=val_loss_history, total_steps=step, num_steps_target=end_step)
                last_save_time = time()
            end

            if time() - last_save_time > 600
                save_checkpoint("checkpoints/checkpoint_latest.json", state_dict, param_keys, uchars, hyperparams;
                    adam_m=adam_m, adam_v=adam_v, step=step, lr=lr, b1=b1, b2=b2,
                    best_val_loss=best_val_loss, train_losses=train_loss_history,
                    val_losses=val_loss_history, total_steps=step, num_steps_target=end_step)
                last_save_time = time()
                println("  [auto-save at step $step]")
            end
        end
    catch e
        if e isa InterruptException
            println("\nTraining interrupted at step $completed_steps!")
        else
            println("\nTraining error at step $completed_steps: $e")
        end
        save_checkpoint("checkpoints/checkpoint_interrupted.json", state_dict, param_keys, uchars, hyperparams;
            adam_m=adam_m, adam_v=adam_v, step=completed_steps, lr=lr, b1=b1, b2=b2,
            best_val_loss=best_val_loss, train_losses=train_loss_history,
            val_losses=val_loss_history, total_steps=completed_steps, num_steps_target=end_step)
        e isa InterruptException || rethrow(e)
    end

    @printf("\nTraining complete in %.1f seconds\n", time() - t_start)
    return best_val_loss, train_loss_history, val_loss_history, completed_steps
end

# ═══════════════════════════════════════════════════════════════
# 10. Run Training
# ═══════════════════════════════════════════════════════════════

lr, b1, b2, eps = 0.01, 0.85, 0.99, 1e-8

adam_m = Dict{String, Matrix{Float32}}(k => zeros(Float32, size(value(state_dict[k]))) for k in param_keys)
adam_v = Dict{String, Matrix{Float32}}(k => zeros(Float32, size(value(state_dict[k]))) for k in param_keys)

mkpath("checkpoints")

NUM_EPOCHS = 3
num_steps = clamp(NUM_EPOCHS * length(train_tokens), 1000, 50000)

println("\n" * "=" ^ 50)
println("Starting training: $num_steps steps, $(length(train_tokens)) sequences")
println("=" ^ 50)

best_val_loss, train_loss_history, val_loss_history, final_step = train_loop!(
    state_dict, params, param_keys, train_tokens, val_tokens,
    adam_m, adam_v, uchars, hyperparams;
    num_steps=num_steps, lr=lr, b1=b1, b2=b2, eps=eps,
    n_layer=n_layer, n_head=n_head, head_dim=head_dim, n_embd=n_embd,
    block_size=block_size, BOS=BOS, gc_interval=GC_INTERVAL)

# Final save
save_checkpoint("checkpoints/final_model.json", state_dict, param_keys, uchars, hyperparams;
    adam_m=adam_m, adam_v=adam_v, step=final_step, lr=lr, b1=b1, b2=b2,
    best_val_loss=best_val_loss, train_losses=train_loss_history,
    val_losses=val_loss_history, total_steps=final_step, num_steps_target=num_steps)

# ═══════════════════════════════════════════════════════════════
# 11. Inference Samples
# ═══════════════════════════════════════════════════════════════

function generate_text(params, uchars, BOS, n_layer, n_head, head_dim, n_embd, block_size;
                       temperature=0.8, max_tokens=128)
    kv_km, kv_vm, kv_l = alloc_kv_cache(n_layer, n_embd, block_size)
    token_id = BOS; sample = Char[]
    for pos in 1:min(max_tokens, block_size)
        logits = gpt(token_id, pos, kv_km, kv_vm, kv_l, params, n_layer, n_head, head_dim)
        probs = Float64.(value(softmax_ag(logits ./ temperature)))
        r = rand(); cum = 0.0; token_id = 1
        for (idx, w) in enumerate(probs)
            cum += w
            if r <= cum; token_id = idx; break; end
        end
        token_id == BOS && break
        push!(sample, uchars[token_id])
    end
    return String(sample)
end

println("\n--- Inference samples ---")
for i in 1:10
    text = generate_text(params, uchars, BOS, n_layer, n_head, head_dim, n_embd, block_size)
    @printf("sample %2d: %s\n", i, text)
end

println("\nDone! Best val loss: $(@sprintf("%.4f", best_val_loss))")
