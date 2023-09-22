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
library(tibble)
library(htmlwidgets)

server <- function(input, output){
  
  # Navbar ------------------------------------------------------------------
  shinyjs::addClass(id = "navBar", class = "navbar-right")
  
  # DT Options --------------------------------------------------------------
  options(DT.options = list( lengthMenu = c(10, 20),
                             dom = 'tl'
  ))  # table and lengthMenu options
  
  
  output$WP <- renderForceNetwork({
    thr <- input$Thr
    ndWP <- input$Node_WP
    
    nodes2$size2 <- nodes2$size/(max(nodes2$size)/28)
    links2 <- links2[links2$value > thr, ]
    links2$value2 <- links2$value/(max(links2$value)/10)
    
    links3 <- data.frame(source = match(links2$source, nodes2$Names) - 1,
                         target = match(links2$target, nodes2$Names) - 1,
                         value = links2$value,
                         value2 = links2$value2)
    if (ndWP == "All"){
      links_vis <- links3
    }
    else {
      id_ndWP <- which(nodes2$Names == ndWP)-1
      links_vis <- links3[links3$source == id_ndWP | links3$target == id_ndWP, ]
    }
    
    script <- 'alert("Name: " + d.Names + "\\n"  +
                     "Prediction acc:" + d.size);'
    
    fn = forceNetwork(Links = links_vis, Nodes = nodes2,
                 Source = "source", Target = "target",
                 Value = "value", NodeID = "Names", linkWidth = JS("function(d) { return Math.sqrt(d.value); }"),
                 Nodesize = "size2", Group = "Groups", radiusCalculation = JS("Math.sqrt(d.nodesize)+6"),
                 opacity = 0.9, zoom = TRUE, fontSize = 10, arrows = TRUE, legend = TRUE, bounded = TRUE, clickAction = script, 
                 colourScale = JS('d3.scaleOrdinal().domain(["strength", "local efficiency", "betweenness centrality", "eigenvector centrality", "global efficiency", "modularity", "participation coefficient", "clustering"]).
                                  range(["#FF0000", " #00FF00", "#0000FF", "#FFFF00", "#800080", "#FFA500", "#FFC0CB", "#A52A2A"])'))
    
    fn$x$nodes$Names <- nodes2$Names
    fn$x$nodes$size <- nodes2$size
    
    htmlwidgets::onRender(fn, jsCode = '
    function (el, x) {
          d3.select("svg").append("g").attr("id", "legend-layer");
          var legend_layer = d3.select("#legend-layer");
          d3.selectAll(".legend")
            .each(function() { legend_layer.append(() => this); });
          d3.select(el)
            .selectAll(".link")
            .append("title")
            .text(d => d.value);
          var link = d3.selectAll(".link")
          var node = d3.selectAll(".node")
      
          var options = { opacity: 0.7,
                          clickTextSize: 10,
                          opacityNoHover: 0.1,
                          radiusCalculation: "Math.sqrt(d.nodesize)+6"
                        }
      
          var unfocusDivisor = 4;
      
          var links = HTMLWidgets.dataframeToD3(x.links);
          var linkedByIndex = {};
      
          links.forEach(function(d) {
            linkedByIndex[d.source + "," + d.target] = 1;
            linkedByIndex[d.target + "," + d.source] = 1;
          });
      
          function neighboring(a, b) {
            return linkedByIndex[a.index + "," + b.index];
          }
      
          function nodeSize(d) {
                  if(options.nodesize){
                          return eval(options.radiusCalculation);
                  }else{
                          return 6}
          }
      
          function mouseover(d) {
            var unfocusDivisor = 4;
      
            link.transition().duration(200)
              .style("opacity", function(l) { return d != l.source && d != l.target ? +options.opacity / unfocusDivisor : +options.opacity });
      
            node.transition().duration(200)
              .style("opacity", function(o) { return d.index == o.index || neighboring(d, o) ? +options.opacity : +options.opacity / unfocusDivisor; });
      
            d3.select(this).select("circle").transition()
              .duration(750)
              .attr("r", function(d){return nodeSize(d)+5;});
      
            node.select("text").transition()
              .duration(750)
              .attr("x", 13)
              .style("stroke-width", ".5px")
              .style("font-size", "18px")
              .style("opacity", function(o) { return d.index == o.index || neighboring(d, o) ? 1 : 0; });
          }
      
          function mouseout() {
            node.style("opacity", +options.opacity);
            link.style("opacity", +options.opacity);
      
            d3.select(this).select("circle").transition()
              .duration(750)
              .attr("r", function(d){return nodeSize(d);});
            node.select("text").transition()
              .duration(1250)
              .attr("x", 0)
              .style("font", options.fontSize + "px ")
              .style("opacity", 0);
          }
      
          d3.selectAll(".node").on("mouseover", mouseover).on("mouseout", mouseout);
            
      }')
    
    
  })
  
  
  
}