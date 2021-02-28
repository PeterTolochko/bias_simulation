using CSV
using DataFrames
using Statistics
using StatsPlots
using TextAnalysis
using Languages
using LinearAlgebra
using Conda
using PyCall
using Distances


# ENV["PYCALL_JL_RUNTIME_PYTHON"] = "/Users/petrotolochko/anaconda3/bin/python"
# ENV["PYTHON"] = "/Users/petrotolochko/anaconda3/bin/python"
# Conda.add("spacy")

# np = pyimport("numpy")
# spacy = pyimport("spacy")

#import en_core_web_sm



function preprocess_text(text)
	text = replace(text, r"[^ a-zA-Z0-9_]" => "")
	text = lowercase(text)
	return text
end



function construct_dtm(text₁, text₂)
	output_array = vcat(text₁, text₂)
	crps = Corpus([TokenDocument(x) for x in output_array])
	languages!(crps, [Languages.German() for x in 1:size(output_array)[1]])
	remove_frequent_terms!(crps)
	prepare!(crps, strip_punctuation)
	update_lexicon!(crps)
	m = DocumentTermMatrix(crps, lexicon(crps))
	my_dtm = dtm(m, :dense)
	return my_dtm, m
end

function fighting_words(text₁, text₂; α=.01, preprocess=false, comparison=true)
	"""Takes two arrays, each for a separate text category
	Outputs ζ̂ scores (for logged odds with Dirichlet prior)"""
	if preprocess
		text₁ = [preprocess_text(text) for text in text₁]
		text₂ = [preprocess_text(text) for text in text₂]
	end
	my_dtm, m = construct_dtm(text₁, text₂)
	vocab_size = length(m.terms)
	# If using flat priors
	if typeof(α) == Float64 || typeof(α) == Int64
		priors = [α for i ∈ 1:vocab_size]
	else
		priors = α
	end
	ζ̂_scores = zeros(vocab_size)
	count_matrix = zeros(2, vocab_size)
	count_matrix[1, :] = sum(my_dtm[1:length(text₁), :], dims=1)
	count_matrix[2, :] = sum(my_dtm[length(text₁):end, :], dims=1)
	α₀ = sum(priors)
	terms = m.terms
	# for future
	# count_matrix[1:end .!=1, i]
	if comparison
		println("Comparing two types; Obtaining Δζ̂...\n") 
		n₁ = sum(count_matrix[1, :])
		n₂ = sum(count_matrix[2, :])
		for i ∈ 1:vocab_size
			y_i = count_matrix[1, i]
			y_j = count_matrix[2, i]
			term₁ = log((y_i + priors[i]) / (n₁ + α₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₂ + α₀ - y_j - priors[i]))
			δ̂ = term₁ - term₂
			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])
			ζ̂_scores[i] = δ̂ / sqrt(σ²)
		end
	else
		println("\nObtaining ζ̂...\n")
		n₁ = sum(count_matrix[1, :])
		n₀ = sum(count_matrix, dims=(1, 2))[1] # total words in the sample
		for i ∈ 1:vocab_size
			y_i = count_matrix[1, i]
			y_j = y_i + count_matrix[2, i]
			term₁ = log((y_i + priors[i]) / (n₁ + α₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₀ + α₀ - y_j - priors[i]))
			δ̂ = term₁ - term₂
			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])
			ζ̂_scores[i] = δ̂ / sqrt(σ²)
		end
	end
	sorted_indices = sortperm(ζ̂_scores)
	return_list = Array{Any, 1}(undef, vocab_size)
	for (index, value) ∈ enumerate(sorted_indices)
		return_list[index] = (terms[value], ζ̂_scores[value])
	end
	return return_list
end



remove_header = r"^(.*?)\(*\) - "


data_path = "data/"


# media = CSV.File(joinpath(data_path, "media_2013.csv")) |> DataFrame
# ots   = CSV.File(joinpath(data_path, "ots_2013.csv")) |> DataFrame

full_data = CSV.File(joinpath(data_path, "with_topics.csv")) |> DataFrame
full_data[!, "unigrams"] = [preprocess_text(x) for x in full_data[!, "unigrams"]]
# full_data = filter(row -> row[:topic] .== 5, full_data)
to_remove = [3, 12, 6, 20]

full_data = filter(row -> row[:topic] ∉ to_remove, full_data)
# full_data = filter(row -> row[:year] .== 2013, full_data)

# ots[!, "text"] = [replace(x, remove_header => "") for x in ots[!, "text"]]


# media[!, "unigrams"] = [preprocess_text(x) for x in media[!, "unigrams"]]
# ots[!, "unigrams"] = [preprocess_text(x) for x in ots[!, "unigrams"]]

# media[!, "bigrams"] = [preprocess_text(x) for x in media[!, "bigrams"]]
# ots[!, "bigrams"] = [preprocess_text(x) for x in ots[!, "bigrams"]]

function get_category_words(category, data, type)

	cat_data = filter(row -> row[:source] .== category, data)
	non_cat_data = filter(row -> row[:source] .!= category && row[:type] .== type, data)

	text₁ = cat_data[!, "unigrams"]
	text₂ = non_cat_data[!, "unigrams"]

	zeta_cat = fighting_words(text₁, text₂; α=.1, comparison=false)

	cat_words = []
	for i in zeta_cat
		if i[2] > 1.6 # 1SD from the mean
			push!(cat_words, i[1])
		end
	end
	return cat_words
end



fpo_words = get_category_words("FPÖ", full_data, "ots")
# spo_words = get_category_words("SPÖ", full_data)


# presse_words = get_category_words("Die Presse", full_data)
# standard_words = get_category_words("Der Standard", full_data)
# heute_words = get_category_words("Heute", full_data)





# fpo_data = filter(row -> row[:source] .== "FPÖ", full_data)
# spo_data = filter(row -> row[:source] .== "SPÖ", full_data)
# presse_data = filter(row -> row[:source] .== "Die Presse", full_data)
# standard_data = filter(row -> row[:source] .== "Der Standard", full_data)
# heute_data = filter(row -> row[:source] .== "Heute", full_data)





# zeta_fpo = fighting_words(text₁, text₂; α=1, comparison=false)

function get_cosine(text₁, text₂, party_words)
	output_array = vcat(text₁, text₂)
	crps = Corpus([TokenDocument(x) for x in output_array])
	languages!(crps, [Languages.German() for x in 1:size(output_array)[1]])
	remove_frequent_terms!(crps, .9)
	prepare!(crps, strip_punctuation)

	update_lexicon!(crps)
	update_inverse_index!(crps)
	m = DocumentTermMatrix(crps, lexicon(crps))
	my_dtm = dtm(m, :dense)

	#my_dtm = dtm(DocumentTermMatrix(crps), :dense)
	flat_dtm_1 = sum(my_dtm[1:length(text₁), :], dims=1)
	flat_dtm_2 = sum(my_dtm[length(text₁):end, :], dims=1)
	flat_dtm = vcat(flat_dtm_1, flat_dtm_2)

	selected_words = party_words
	indices = []
	for (i, word) ∈ enumerate(m.terms)
		if word ∈ selected_words
		# println(word, " -> ", i)
			push!(indices, i)
		end
	end

	numerator = flat_dtm[1, indices] ⋅ flat_dtm[2, indices]
	denominator = sqrt(flat_dtm[1, indices] ⋅ flat_dtm[1, indices]) * sqrt(flat_dtm[2, indices] ⋅ flat_dtm[2, indices])
	cosine_sim = numerator/denominator

	return cosine_sim

end


parties = ["BZÖ"
 		  "FPÖ"
 		  "SPÖ"
          "ÖVP"
          "GRÜNE"
          "NEOS"]

media = [
"Heute"
"Die Presse"
"Der Standard"
"Falter"
"Kurier"
"Österreich"
"Kronen Zeitung"
"Kleine Zeitung"]

important = union(parties, media)
full_data = filter(row -> row[:source] ∈ important, data)


Sims = zeros(length(parties), length(media))

for (i, party) ∈ enumerate(parties), (j, medium) ∈ enumerate(media)
	println("Comparing $(party) with $(medium)...")
	party_data = filter(row -> row[:source] .== party, full_data)
	medium_data = filter(row -> row[:source] .== medium, full_data)

	party_words = get_category_words(party, full_data, "ots")
	# medium_words = get_category_words(party, full_data)

	sim = get_cosine(party_data[!, "unigrams"], medium_data[!, "unigrams"],
	party_words)

	Sims[i, j] = sim
end

Sims ./ mean(Sims, dims=2)

get_cosine(spo_data[!, "unigrams"], heute_data[!, "unigrams"],
	spo_words, heute_words)

# fpo/presse 0.22658328139053485
# spo/presse 0.36015558122911945
# fpo/standard 0.15912946381935625
# spo/standard 0.18945312038785875
# fpo/heute 0.16227887049760847
# spo/heute 0.10971247931341756

