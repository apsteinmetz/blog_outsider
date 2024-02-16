t_tidy <- function(df, rowlabel_col_name = "item"){
  if(has_rownames(df)){
    df <- rownames_to_column(df, var = "V0")
  }
  df <- as_tibble(df)
  if(length(unique(sapply(df[,-1], class))) > 1){
    warning("There are a mix of data types in the data columns.  They will be coerced to character.\n")
    df <- df |> mutate(across(everything(), as.character))
  }
  if(rowlabel_col_name %in% pull(df[,1])){
    warning("The rowname column name is a duplicate of another column name.  Changing\n")
    rowlabel_col_name <- paste0(rowlabel_col_name,"..2")
  }

  df |>
    pivot_longer(cols = -1, names_to = rowlabel_col_name, values_to = "value",names_repair = "unique") |>
    rename_with(~ rowlabel_col_name, .cols = 2) |>
    pivot_wider(names_from = 1, values_from = value)
}

rowlabel_col_name <- "model"
cars_df |> t_tidy(rowlabel_col_name) |> t_tidy()
cars |> t_tidy("item")

library(topicmodels)
library(tidytext)
data("AssociatedPress")
ap <- AssociatedPress |> tidy() |>
  pivot_wider(names_from = document, values_from = count)


microbenchmark::microbenchmark(times = 1,
  ap_t <- ap |>  t_dt("doc"),
  ap_t <- ap |>  t_df("doc"),
  ap_t <- ap |>  t_tidy("doc")
)

