let svgContent = d3.select("#content")
  .append("svg")
  .attr("width", 400)
  .attr("height", 400);

// Draw rectangle
svgContent.append("rect")
  .attr("x", 10)
  .attr("y", 10)
  .attr("width", 100)
  .attr("height", 100);
