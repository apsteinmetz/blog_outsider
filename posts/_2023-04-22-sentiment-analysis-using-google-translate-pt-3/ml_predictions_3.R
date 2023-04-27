# modeling of sentiment
library(tidyverse)
library(tidytext)
library(quanteda)
library(tictoc)
library(ranger)
library(xgboost)

#setwd("2023-02-28_african_language")
load("~/R Projects/tidytuesday/2023-02-28_african_language/data/stopwords_af.rdata")
load("~/R Projects/tidytuesday/2023-02-28_african_language/data/afrisenti_translated.rdata")



# ----- SETUP ------------------------------
afrisenti_translated <- afrisenti_translated |>
  mutate(lang = as.factor(assigned_long)) |>
  # make binary
  # filter(label != "neutral") |> 
  mutate(sentiment = as.factor(as.character(label))) |> 
  select(-label)
  

# make negation tokens
afrisenti_translated <- afrisenti_translated |>
  mutate(tweet = str_replace(tweet, "not ","not_")) |>
  mutate(translatedText = str_replace(translatedText, "not ","not_"))


tweet_train <- afrisenti_translated |> 
  filter(intended_use == "train") |> 
  select(tweet_num,sentiment,lang,tweet,translatedText)

tweet_test <- afrisenti_translated |> 
  filter(intended_use == "test") |> 
  select(tweet_num,sentiment,lang,tweet,translatedText)

tweet_dev <- afrisenti_translated |> 
  filter(intended_use == "dev") |> 
  select(tweet_num,sentiment,lang,tweet,translatedText)

# add my stop words to defaults
my_stop_words = tibble(word = c("http","https","dey","de","al","url","na","t.co","rt","user","users","wey","don",
                                as.character(1:100)))
                           

# make a stopword list of any 1-character words
# this is a somewhat arbitrary rubric for African language stop words
stop_words_1char <- afrisenti_translated |> 
  unnest_tokens(word,tweet) |> 
  select(word) |> 
  filter(str_length(word)<2) |> 
  unique()

full_stop_words <-  c(
  stop_words$word,
  my_stop_words$word,
  stopwords_af$word,
  stop_words_1char$word
)

# -------------------------------------------
# non-tidymodel preproccessing functions

# turn words preceded by "not" into "not_<word>"
# to create a negated token
detect_negations <- function(tokens,negation_words = c("not")) {
  # function to negate tokenized data
  tokens <- tokens |> rowid_to_column(var="word_num")
  not_words_rows <- tokens |> 
    filter(word %in% negation_words) |> 
    mutate(word_num = word_num) |> 
    pull(word_num)
  tokens <- tokens |> 
    # create negated terms
    filter(!(word_num %in% not_words_rows)) |> 
    mutate(word = ifelse(word_num %in% (not_words_rows+1),paste0("not_",word),word)) |> 
    select(-word_num)
  return(tokens)
}

only_top_words <- function(tokens, word_count = 1000) {
  chosen_words <- tokens |>
    ungroup() |>
    select(word) |>
    count(word) |>
    slice_max(order_by = n, n = word_count) |> 
    select(-n)
  return(inner_join(tokens,chosen_words))
}

token_filter <- function(tokens) {
  tokens |>
    # create negations where "not" is before a word
    detect_negations() |> 
    # remove English stop words
    anti_join(stop_words) |>
    # remove African stop words by language
    anti_join(stopwords_af) |>
    # remove my additional stop words
    anti_join(my_stop_words) |>
    # call any word of 1 or 2 characters a stop word
    filter(str_length(word) > 2)
}
# --------------------------------------
# make sparse document-feature matrix

make_dfm <- function(tweet_data,translated = FALSE,num_words = 1000) {
  if(translated){
    tweet_tokens <- unnest_tokens(select(tweet_data,tweet_num,translatedText),word, translatedText)
  } else{
    tweet_tokens <- unnest_tokens(select(tweet_data,tweet_num,tweet),word,tweet)
  } 
tweet_tokens <- tweet_tokens |> 
  token_filter() |>
  only_top_words(num_words) |>
  count(tweet_num, word) |>
  bind_tf_idf(word, tweet_num, n) |>
  select(tweet_num, word, tf_idf)

sentiment_subset <- tweet_tokens |> 
  slice_head(n=1,by=tweet_num) |> 
  left_join(tweet_data) |> 
  pull(sentiment)

tweet_dfm <- cast_dfm(tweet_tokens,tweet_num,word,tf_idf)
# add dependent variable to sparse matrix
docvars(tweet_dfm,"sentiment") <- sentiment_subset

return(tweet_dfm)
}


# more words in common in the translated word list
translated = FALSE
tweet_train_dfm <- make_dfm(tweet_train,translated = translated,num_words = 2000)
tweet_test_dfm <- make_dfm(tweet_test,translated = translated,num_words = 2000)


# how close are the word lists?
# more words in common in the translated word list
inner_join(enframe(dimnames(tweet_train_dfm)[[2]]),
           enframe(dimnames(tweet_test_dfm)[[2]]),
           by = "value") |> nrow() |> paste("Words in both train and test sets")

# make sure test set has all variables in both train and test sets
tweet_test_dfm <- dfm_match(tweet_test_dfm, 
                      features = featnames(tweet_train_dfm))

# -------------------------------------------
# run the models

tic()
rf_fit <- ranger::ranger(y= tweet_train_dfm$sentiment,
                         x=tweet_train_dfm,
                         num.trees = 100,
                         classification = TRUE)

predictions <- predict(rf_fit,tweet_test_dfm)
toc()

# Validation set assessment #1: looking at confusion matrix
predicted_for_table <- tibble(actual = tweet_test_dfm$sentiment,
                              predictions$predictions)


caret::confusionMatrix(table(predicted_for_table))

tic()
xg_fit <- xgboost(
  data = tweet_train_dfm,
  max.depth = 100,
  nrounds = 100,
  objective = "multi:softmax",
  num_class = 3,
  label = as.numeric(tweet_train_dfm$sentiment)-1,
  print_every_n = 10
)
toc()

xg_fit$evaluation_log |> 
  ggplot(aes(iter,train_mlogloss)) + geom_line()

predicted <- predict(xg_fit,tweet_test_dfm) |> 
  as.factor()

levels(predicted) <- levels(tweet_test$sentiment)
predicted_for_table <- tibble(actual = tweet_test_dfm$sentiment,
                              predicted)


caret::confusionMatrix(table(predicted_for_table))

