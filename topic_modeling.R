require(stm)
require(tidyverse)


data_ots_13 <- read_csv('~/Desktop/bias_simulation/data/ots_2013.csv')
data_ots_17 <- read_csv('~/Desktop/bias_simulation/data/ots_2017.csv')
data_media_13 <- read_csv('~/Desktop/bias_simulation/data/media_2013.csv')
data_media_17 <- read_csv('~/Desktop/bias_simulation/data/media_2017.csv')

data_ots_13$year <- "2013"
data_ots_17$year <- "2017"
data_media_13$year <- "2013"
data_media_17$year <- "2017"

data_ots_13$type <- "ots"
data_ots_17$type <- "ots"
data_media_13$type <- "media"
data_media_17$type <- "media"

full_data <- bind_rows(data_ots_13,
                       data_ots_17,
                       data_media_13,
                       data_media_17)


ots_data <- full_data %>%
  filter(type == "ots")
media_data <- full_data %>%
  filter(type == "media")


processed <- textProcessor(ots_data$bigrams, metadata=ots_data,
                           removepunctuation = FALSE,
                           customstopwords = c("wien",
                                               "wiener",
                                               "stadtwien",
                                               "österreich",
                                               "kärnten",
                                               "österreicher",
                                               "österreichisch",
                                               "pölten"),
                           custompunctuation = ".")

out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

docs <- out$documents
vocab <- out$vocab
meta <- out$meta




my_model <- stm(out$documents, out$vocab, K=60, 
                       max.em.its=75, data=out$meta, init.type="Spectral", 
                prevalence = ~year,
                       seed=8458159)
plot(my_model, n=10)



processed_media <- textProcessor(media_data$bigrams, metadata=media_data,
                           removepunctuation = FALSE,
                           customstopwords = c("wien",
                                               "wiener",
                                               "stadtwien",
                                               "österreich",
                                               "kärnten",
                                               "österreicher",
                                               "österreichisch",
                                               "pölten"),
                           custompunctuation = ".")

# out_media <- prepDocuments(processed$documents, processed$vocab, processed$meta)

newdocs <- alignCorpus(new=processed_media, old.vocab=vocab)


new_model <- fitNewDocuments(model=my_model, documents=newdocs$documents, newData=newdocs$meta,
                origData=out$meta, prevalence=~year,
                prevalencePrior="Covariate")



max_topic_ots <- apply(my_model$theta, 1, which.max)
max_topic_media <- apply(new_model$theta, 1, which.max)

# 
# full_data$topic <- max_topic
# 
# garbage_topics <- c(20, 6, 12, 3)

out_data_ots <- tibble(
  source = processed$meta$source,
  unigrams = processed$meta$unigrams,
  bigrams = newdocs$meta$bigrams,
  year = processed$meta$year,
  type = processed$meta$type,
  topic = max_topic_ots
)

out_data_ots$source[out_data_ots$source == "Grüne"] <- "GRÜNE"
out_data_ots$source[out_data_ots$source == "Team FRANK"] <- "FRANK"
out_data_ots %>% count(source)



out_data_ots %>% count(source)

out_data_media <- tibble(
  source = newdocs$meta$source,
  unigrams = newdocs$meta$unigrams,
  bigrams = newdocs$meta$bigrams,
  year = newdocs$meta$year,
  type = newdocs$meta$type,
  topic = max_topic_media
)
out_data_media %>% count(source) %>% print(n = nrow(.))


out_data_media$source[with(out_data_media, grepl("ATV", source))] <- "ATV"
out_data_media$source[with(out_data_media, grepl("derStandard.at", source))] <- "derstandard.at"
out_data_media$source[with(out_data_media, grepl("Salzburger Nachrichten", source))] <- "Salzburger Nachrichten"
out_data_media$source[with(out_data_media, grepl("Die Presse", source))] <- "Die Presse"
out_data_media$source[with(out_data_media, grepl("Heute", source))] <- "Heute"
out_data_media$source[with(out_data_media, grepl("Krone.at", source))] <- "krone.at"
out_data_media$source[with(out_data_media, grepl("Kurier.at", source))] <- "kurier.at"
out_data_media$source[with(out_data_media, grepl("Oe24.at", source))] <- "oe24.at"
out_data_media$source[with(out_data_media, grepl("Tiroler Tageszeitung", source))] <- "Tiroler Tageszeitung"
out_data_media$source[with(out_data_media, grepl("WOCHE", source))] <- "WOCHE"
out_data_media$source[with(out_data_media, grepl("ZIB", source))] <- "ZIB"
out_data_media$source[with(out_data_media, grepl("Zeit im Bild ", source))] <- "ZIB"








topics_table <- table(out_data_ots$source, out_data_ots$topic)
chisq <- chisq.test(topics_table, correct = FALSE)
chisq$observed

round(chisq$expected, 2)


round(topics_table / rowSums(topics_table), 2)
column_names <- c("Topic", colnames(t(topics_table)))
df_table <- as.data.frame(t(topics_table))

topics_table <- as.matrix(topics_table)
as_data_frame(topics_table)


t(topics_table)

save.image("topics_image.RData")

out_data <- bind_rows(out_data_ots, out_data_media)

# out_data <- out_data %>%
#   filter(!(topic %in% garbage_topics))
write_csv(out_data, "~/Desktop/bias_simulation/data/with_topics.csv")
