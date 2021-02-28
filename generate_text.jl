using Random
using Distributions
using TextAnalysis
using LinearAlgebra
using CSV
using DataFrames
using Statistics
using StatsBase

### Possibly Check for Brown Corpus word frequencies
### and sample with the same probability

# read CSV with word frequencies from H.G. Wells Time Machine
frequency_table = DataFrame(CSV.File("frequency_table.csv"))

# probabilities here should be a factor of ncol(agenda_words)
# higher to correct for random sampling in the function
agenda_p = .05

agenda_words = [
"1111" .01 agenda_p
"2222" .01 agenda_p
"3333" .01 agenda_p
"4444" .01 .01
"5555" .01 .01
"6666" .01 .01
"7777" .01 .01
"8888" agenda_p .01
"9999" agenda_p .01
"0000" agenda_p .01]


# correct for later random sampling out of ncol(agenda_words)
agenda_words[:, 2:end] = agenda_words[:, 2:end] .* size(agenda_words)[1]


function generate_text(t_length, agenda_type)

	# determine nrows of agenda matrix
	agenda_size = size(agenda_words)[1]

	# add 1, since agenda_type is agenda_words column index for probabilities
	# starting from column 2
	agenda_type = agenda_type + 1

	# generate random "text" of size t_length
	# and with vocabulary = vocab
	# random_text = sample(1.:vocab, t_length; replace=true)
	random_text = sample(frequency_table[!, 1], Weights(frequency_table[!, 2]), t_length)

	# loop over the generated "text"
	# each entry has an equal change to become an "agenda word"
	for i in 1:length(random_text)
		# randomly select an "agenda word"
		agenda = sample(1:agenda_size, 1)

		# the word i becomes an "agenda word"
		# with probability p = agenda_words[agenda, 2]
		p = agenda_words[agenda, agenda_type][1] # pull the float from the array
		if rand(Bernoulli(p))
			random_text[i] = agenda_words[agenda, 1][1]
		end
	end
	return random_text
end

# test = generate_text(1000, 1)
# sum(test .== 0000)



function get_flat_corpus(t_length, corpus_size=1000)
	output_array_1 =  Array{Array{SubString{String},1},1}(undef, corpus_size)
	output_array_2 =  Array{Array{SubString{String},1},1}(undef, corpus_size)

	for i in 1:corpus_size
		output_array_1[i] = generate_text(t_length, 1)
		output_array_2[i] = generate_text(t_length, 2)
	end
	output_array = vcat(output_array_1, output_array_2)
	crps = Corpus([TokenDocument(x) for x in output_array])
	update_lexicon!(crps)

	my_dtm = dtm(DocumentTermMatrix(crps), :dense)
	flat_dtm_1 = sum(my_dtm[1:1000, :], dims=1)
	flat_dtm_2 = sum(my_dtm[1001:2000, :], dims=1)
	flat_dtm = vcat(flat_dtm_1, flat_dtm_2)
	return flat_dtm
end

flat_1 = get_flat_corpus(1000)

# dot(flat_1[1, :], flat_1[2, :])

numerator = flat_1[1, :] ⋅ flat_1[2, :]
denominator = sqrt(flat_1[1, :] ⋅ flat_1[1, :]) * sqrt(flat_1[2, :] ⋅ flat_1[2, :])
numerator/denominator
# 0.4561719395471122

