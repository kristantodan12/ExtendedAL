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
                         windowTitle = "Visualization of Multiverse Analysis",
                         position = "fixed-top",
                         header = tags$style(
                           ".navbar-right {
                       float: right !important;
                       }",
                           "body {padding-top: 150px;}"),
                         
                         tabPanel("Visualization of Multiverse Analysis", value = "WH",
                                  sidebarLayout( 
                                    
                                    sidebarPanel( width = 3,
                                                  introjsUI(),
                                                  
                                                  useShinyjs(),
                                                  
                                                  tags$div(
                                                    style = "height:50px;",
                                                    introBox(
                                                      tags$div(
                                                        style = "height:30px;",
                                                        actionLink("settings", "Options", 
                                                                   icon = icon("sliders", class = "fa-2x"))),
                                                      data.step = 6,
                                                      data.intro = "Set your settings and preferences."
                                                    ),
                                                    selectInput("Node_WP",
                                                                label   = "Select the node you want to explore the connections",
                                                                choices =  c("All", nodes$Names),
                                                                selected = "All"
                                                    ),
                                                    sliderInput("Thr", "Threshold connection (correlation between forking paths). Please note that setting the threshold to small values lead to dense network which is heavy to deploy",
                                                                min = 0, max = 1,
                                                                value = 0.7
                                                    ),
                                                    
                                                  )
                                    ),  
                                    mainPanel( width = 9,
                                               forceNetworkOutput(outputId = "WP", width = "100%", height = "700px")
                                    )  
                                  )  
                         ), 
                         
                         
)

)