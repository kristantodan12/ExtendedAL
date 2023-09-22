#######
# This code is the helper function to genrate the interactive shiny app.
# It reads the data from python script "GetDataforShiny" of nodes and links dataframe

library(shiny)
library(networkD3)
library(dplyr)
library(igraph)
library(visNetwork)
library(geomnet)
library(stringr)
library(png)
library(shinyjs)
library(DT)
library(rintrojs)
library(ggplot2)
library(qdapTools)
library(RColorBrewer)
rm(list = ls())
Sys.setlocale('LC_CTYPE','C')


###Reading the data
nodes <- read.csv(file = 'nodes.csv')
colnames(nodes) <- c('Names', 'size')
nodes <- nodes %>% 
  mutate(Groups = case_when(grepl('strength', Names) ~ 'strength',
                            grepl('local efficiency', Names) ~ 'local efficiency',
                            grepl('betweennness centrality', Names) ~ 'betweenness centrality', 
                            grepl('eigenvector centrality', Names) ~ 'eigenvector centrality',
                            grepl('global efficiency', Names) ~ 'global efficiency',
                            grepl('modularity', Names) ~ 'modularity',
                            grepl('participation coefficient', Names) ~ 'participation coefficient',
                            grepl('clustering', Names) ~ 'clustering', 
                            TRUE ~ NA_character_
                          ))

links <- read.csv(file = 'links.csv')
links <- subset(links, select = -X) #Remove first column
links$source<-gsub("'","",as.character(links$source))
links$source<-gsub("\\[|\\]","",as.character(links$source))
links$target<-gsub("'","",as.character(links$target))
links$target<-gsub("\\[|\\]","",as.character(links$target))

nodes$size <- abs(nodes$size)
links$value <- abs(links$value)
nodes2 <- nodes
links2 <- links