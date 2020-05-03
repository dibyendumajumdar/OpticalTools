let svgContent = d3.select("#content")
  .append("svg")
  .attr("width", 400)
  .attr("height", 400);

let line = d3.line()([[10, 60], [40, 90], [60, 10], [190, 10]]);
svgContent.append("path")
  .attr("d", line)
  .attr("stroke", "blue")
  .attr("fill", "none");
