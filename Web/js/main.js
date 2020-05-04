// Leitz summicron 50mm f2
function GetData() {
  return [
    {id:1, radius:42.71, thickness:3.99, index:1.73430, diameter:28.94, vd:28.19},
    {id:2, radius:195.38,	thickness:0.20, diameter:27.06 },
    {id:3, radius:20.50, thickness:7.18, index:1.67133, diameter:24.02, vd:41.64},
    {id:4,	radius:0, thickness:1.29, index:1.79190, diameter:21.49, vd:25.55},
    {id:5, radius:14.94, thickness:5.35, diameter:18.39},
    {id:6, stop:true,	thickness:7.61, diameter:18.059},
    {id:7,	radius:-14.94, thickness:1.00, index:1.65222, diameter:17.50, vd:33.60},
    {id:8, radius:0, thickness:5.22, index:1.79227, diameter:19.27, vd:47.15},
    {id:9, radius:-20.50, thickness:0.20, diameter:20.38},	
    {id:10, radius:0, thickness:3.69, index:1.79227, diameter:22.96, vd:47.15},
    {id:11,	radius:-42.71, thickness:37.32, diameter:23.97}	
  ];
}

// Gets the total length of the system including image plane
function GetLength(data) {
  let len = 0;
  for (let i = 0; i < data.length; i++) {
    len += data[i].thickness;
  }
  return len;
}

// Gets the max diameter amongst all surfaces
function GetMaxDiameter(data) {
  let h = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i].diameter > h)
      h = data[i].diameter;
  }
  return h;
}

// https://en.wikipedia.org/wiki/Circular_segment
function GetAngleOfArc(diameter, chord) {
  return 2 * Math.asin(chord/diameter);
}

let data = GetData();
let len = Math.round(GetLength(data)*1.5);
let angle = GetAngleOfArc(42.71*2, 28.94);
let adjustment = Math.PI + (Math.PI-angle)/2; // The arc angle starts at 12 o'clock position
let startAngle = adjustment;
let endAngle = angle+adjustment;

//let viewport = "0 0 " + 
let svgContent = d3.select("#content")
  .append("svg")
  .attr("width", 1000)
  .attr("height", 1000)
  .attr("viewBox", "0 0 " + len.toString() + " " + len.toString());

let arc = d3.arc()
  .innerRadius(42.71)
  .outerRadius(42.71)
  .startAngle(startAngle)
  .endAngle(endAngle);

let g = svgContent.append("g")
  .attr("transform", "translate(" + Math.round(len/2) + "," + Math.round(len/2) + ")");
g.append("path")
  .attr("d", arc())
  .attr("stroke", "blue")
  .attr("fill", "none")  
  ;