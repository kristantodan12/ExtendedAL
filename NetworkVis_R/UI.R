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
library(shinyWidgets)

shinyWidgets::shinyWidgetsGallery()

source("helper_fun.R")

ui <- shinyUI(navbarPage(title = img(src="metarep.jpg", height = "100px"), id = "navBar",
                         theme = "bootstrap.css",
                         collapsible = TRUE,
                         inverse = TRUE,
                         windowTitle = "Visualisation of Multiverse Analysis",
                         position = "fixed-top",
                         header = tags$style(
                           ".navbar-right {
                       float: right !important;
                       }",
                           "body {padding-top: 150px;}"),
                         
                         tabPanel("Visualisation of Multiverse Analysis", value = "WH",
                                  sidebarLayout( 
                                    
                                    sidebarPanel( width = 3,
                                                  shiny::HTML("<h5><b>Explore the resutls of the multiverse analysis in a network fashion.</b><br><br>
                                                                     Each node (circle) represents a pipeline. The 
                                                                     color of a node indicates the graph measure used by the pipeline. <br><br>
                                                                     Edges are the similarity between pipelines, quantified by the correlation
                                                                     of the graph measures. The wider the arrow, the higher the similarity. <br><br>
                                                                     By hovering over a node you can get the name of the pipeline. If you click 
                                                                     on it you get its definition and the
                                                                     prediction accuracy in terms of explained variance of general cognition
                                                                     by graph measures. By hovering over an edge you can get the similarity between pipelines. <br><br>
                                                                     To identify edges of a specific pipeline, please 
                                                                     select it in the dropdown below. If you choose all, you will 
                                                                     see edges of all pipelines.<br><br>"
                                                  ),
                                                  selectInput("Node_WP",
                                                              label   = "Explore edges of pipeline:",
                                                              choices =  c("All", nodes$Names),
                                                              selected = "All"
                                                  ),
                                                  sliderInput("Thr", "Threshold similarity value",
                                                              min = 0, max = 1,
                                                              value = 0.7
                                                  ),
                                                  shiny::HTML("<h5>Move the threshold to only see edges (similarity between pipelines) higher than the threshold.
                                                              Please note that setting the threshold to small values lead to dense network which is heavy to deploy. </b><br><br>
                                                              For more details about this app, please refer to [link to paper] </h5>"),
                                                  
                                    ),  # Closes sidebarPanel
                                    mainPanel( width = 9,
                                               forceNetworkOutput(outputId = "WP", width = "100%", height = "700px")
                                    )  # Closes the mainPanel
                                  )  # Closes the sidebarLayout
                         ),  # Closes the second tabPanel called "Literature-based Analysis"
                         
                         
)

)