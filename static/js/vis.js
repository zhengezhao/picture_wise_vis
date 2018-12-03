var classes = ['0', '1', '2', '3','4', '5', '6', '7', '8', '9'];
var nn_chosen = 0;//default choice is the last one
var options = [];
var svgWidth =  600;
var svgHeight = 350;
var margin = {top:20, bottom:20, left:30, right:20};
var width = +svgWidth- margin.left-margin.right;
var height = +svgHeight -margin.top- margin.bottom;
var query_data,query_instance_data;

var data_chosen =  data[nn_chosen]['data'];
var data_chosen_pre ;
var x,new_x,y,mousex, invertedx,class_chosen, epoch_chosen;
//true  if the first is drawn
var onfirstbarchart = true;
$("#SPReset").hide();



d3.selection.prototype.moveToFront = function() {
    return this.each(function(){
        this.parentNode.appendChild(this);
      });
};


d3.selection.prototype.callReturn = function(callable)
{
    return callable(this);
};


for (var i = 0; i <= parseInt(parseInt(num_of_nn)); i++) {
    options.push(i);
}

var select = document.getElementById("selectNumber");

for(var i = 0; i < options.length; i++) {
    var opt = options[i];
    var el = document.createElement("option");
    if(opt === 0){
      el.textContent = "AVERAGE";
    }
    else{
      el.textContent = opt;
    }
    el.value = opt;
    //console.log(el);
    select.appendChild(el);

}


var tip = d3.tip()
.attr("class", "d3-tip")
.html(function(d,i){
    //console.log(i);
    var e = i.epoch;
    var k = d;
    var text;
    text = 'EPOCH: ' + e.toString() + "<br/>"+ "CLASS "+"  VALUE     "+ "<br/>";

    for(var j=0; j<10;j++){

        var a =  i[j].toFixed(4);

        if (j != k){
            text+= j.toString()+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+a+"<br/>";
        }
        else{
            text+=  "<span style='color:red'>"+j.toString()+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+a+"</span>"+"<br/>";
        }
    }
    return text;
} );

d3.select("#streamgraphdiv")
  .callReturn(createSvg);

d3.select("#streamgraph").append("defs").append("clipPath")
  .attr("id","clip")
  .append("rect")
  .attr("width",width)
  .attr("height",height);



var svg=d3.select("#streamgraph").append("g")
    .attr("class","inner_space")
    .attr("transform","translate("+margin.left+","+margin.top+")");


svg.call(tip);

var color = d3.scaleOrdinal(d3.schemeCategory10);



function mathMutiply(w1,w2){
  var w1 = math.matrix(w1);
  var w2 = math.matrix(w2);
  var w = math.multiply(w1,w2);
  return w.valueOf();
}


function Matrix_Calculate(o_data,weight,weight_prev,data_origin,data_prev){

  var o_data= math.matrix(o_data);
  var weight = math.matrix(weight);
  var weight_prev = math.matrix(weight_prev);
  console.log("data_origin",data_origin);
  console.log("data_prev",data_prev);
  console.log("o_data",o_data);
  console.log("weight",weight);
  var w = math.multiply(o_data,weight);
  console.log("w",w);
  console.log("weight_prev",weight_prev);
  var aw = math.map(w,function(value,i){
    return value*data_origin[i] ;
  });

  console.log("aw",aw);

  var w_prev = math.multiply(o_data,weight_prev);
  console.log("w_prev",w_prev);
  var aw_prev = math.map(w_prev, function(value,i){
    return value*data_prev[i];
  });
  console.log("aw_prev",aw_prev);

  var result = aw.map(function(item,index){
    // console.log(item,typeof(item));
    // console.log(index,aw_prev[index], typeof(aw_prev[index]));
    return aw = aw_prev.get(index);
  });

  //console.log(result);

  return result.valueOf();
}

// function twodarray(m,n){
//   var x = new Array(m);
//   for(var i=0; i<x.length;i++){
//     x[i] = Array.from({length: n}, (v, i) => 0);
//   }
//   return x;
// }

// function matrixKernel(kernel, matrix){ //kernel :5X5 matrix:64X144
//   var concat_kernel;
//    for(var j=0; j<kernel.length;j++){
//       for(var k =0; k<kernel[j].length;k++){


//       }


//     }



//   for(var i=0; i<matrix.length/2;i++){


//     for(var j=0; j<kernel.length;j++){
//       for(var k =0; k<kernel[j].length;k++){
//         matrix[i][i+j] = kernel[j][k];

//       }
//     }

//   }
// }


// function deconvolution(weight,kernel){
//   var result  = rray.from({length: 1440}, (v, i) => 0);
//   //size of weight: 320:20X4X4
//   //size of the kernel : 20,10,5,5
//   for(var i=0; i<kernel.length;i++){ //kernel.length = 20
//     var row_width = weight.length/kernel.length; //16
//     var output = Array.from({length: row_width*4}, (v, i) => 0); //maxpool :size: 64
//     var origin_output = weight.slice(i*row_width,(i+1)*row_width);
//     var output_row_width = Math.sqrt(row_width)*4/2;
//     for(var j=0; j<origin_output.length;j++){
//       var c = j % Math.sqrt(row_width);
//       var r = Math.floor(j/Math.sqrt(row_width));
//       output[(r*2)*output_row_width+c*2] = origin_output[j]/4;
//       output[(r*2)*output_row_width+c*2+1] = origin_output[j]/4;
//       output[(r*2+1)*output_row_width+c*2] = origin_output[j]/4;
//       output[(r*2+1)*output_row_width+c*2+1] = origin_output[j]/4;
//     }

//     for(var k=0; k<kernel[i].length;k++){ //10 kernels
//       var matrix_kernel = twodarray(output.length,result.length/kernel[i].length); //64X144

//      // kernel[i][k] //5X5
//     }

//   }
// }

function BackPropagation(bar_index, o_data){
  var bar_data_origin = query_instance_data[bar_index].data_origin;
  var bar_data_prev = query_instance_data[bar_index].data_prev;
  var bar_data,bar_data_o, bar_data_p;

  if(bar_index ==2){
      bar_data_o = bar_data_origin.map(function(d,i){return d*o_data[i];});
      bar_data_p = bar_data_prev.map(function(d,i){return d*o_data[i];});
      bar_data = bar_data_o.map(function(d,i){return d- bar_data_p[i]});

      return [bar_data_o , bar_data_p, bar_data];
  }


  else{
    var bar_data_o_t= BackPropagation(bar_index+1,o_data)[0];
    var bar_data_p_t = BackPropagation(bar_index+1,o_data)[1];
    bar_data_o = bar_data_origin.map(function(d,i){return d*mathMutiply(bar_data_o_t,query_instance_data[bar_index+1].weight)[i]});
    bar_data_p = bar_data_prev.map(function(d,i){return d*mathMutiply(bar_data_p_t,query_instance_data[bar_index+1].weight_prev)[i]});
    bar_data = bar_data_o.map(function(d,i){return d- bar_data_p[i]});
    return [bar_data_o , bar_data_p, bar_data];
  }

}

function ImagePrediction(o_data){
  var bar_data_o_t = BackPropagation(0,o_data)[0];
  var bar_data_p_t = BackPropagation(0,o_data)[1];
  var weight_data_o = mathMutiply(bar_data_o_t, query_instance_data[0].weight);
  var weight_data_p = mathMutiply(bar_data_p_t, query_instance_data[0].weight_prev);
  var weight_data = weight_data_o.map(function(d,i){return d - weight_data_p[i];});
  return weight_data;

}

function DrawImageHeatMap(o_data){

  $("#HeatMapDiv").empty();
  var svgWidth = document.getElementById("HeatMapDiv").offsetWidth;
  var svgHeight = document.getElementById("HeatMapDiv").offsetHeight;
  var image_svgWidth = document.getElementById("HeatMapDiv").offsetWidth;
  var image_svgHeight = document.getElementById("HeatMapDiv").offsetHeight;
  var margin = {top:0, bottom:0, left:0, right:0};
  var width = +svgWidth- margin.left-margin.right;
  var height = +svgHeight -margin.top- margin.bottom;
  var data = ImagePrediction(o_data);
  var rows = 28;
  var cols  = 28;

  var colorMap = d3.scaleLinear().domain([d3.min(data),0, d3.max(data)]).range([d3.interpolateRdBu(0),d3.interpolateRdBu(0.5),d3.interpolateRdBu(1)]);

  var x = d3.scaleLinear().domain([0,cols-1]).range([0,width]).nice();
  var y = d3.scaleLinear().domain([0,rows-1]).range([height,0]).nice();



  var svg = d3.select("#HeatMapDiv")
              .append("svg")
              .attr("id","HeatMapSvg")
              .attr("width", svgWidth)
              .attr("height", svgHeight)
              .append("g")
              .attr("transform","translate("+margin.left+","+margin.top+")");



  var heats =  svg.selectAll(".heats")
            .data(data)
            .enter().append("rect")
            .attr("class","heats");


  heats.attr("x", function(d,i) { return x(i % rows); })
        .attr("y", function(d,i) { return y(Math.floor(i / rows)); })
        .attr("width", width/cols )
        .attr("height", height/rows)
        .style("fill",function(d){return colorMap(d);})
        .style("opacity",0.5);
};





function DrawBarCharts(query_instance_data){


  var svgWidth = document.getElementById("BarChartDiv").offsetWidth/3-5;
  var svgHeight = document.getElementById("BarChartDiv").offsetHeight;
  var margin = {top:20, bottom:20, left:30, right:20};
  var width = +svgWidth- margin.left-margin.right;
  var height = +svgHeight -margin.top- margin.bottom;


  var selectedbar=null;
  barchartName = "BarChartDiv";


  var o_data = Array.from({length: 10}, (v, i) => 1);
  DrawImageHeatMap(o_data);

  for(var bar_i =0; bar_i<=2; bar_i++){
    var data_id = query_instance_data[bar_i].label;

    var bar_data = BackPropagation(bar_i, o_data)[2];


    //console.log(data_id,bar_data);
    var g = d3.select('#'+barchartName)
              .append("svg")
              .attr("id",data_id)
              .attr("width", svgWidth)
              .attr("height", svgHeight)
              .append("g")
              .attr("transform","translate("+margin.left+","+margin.top+")");

        //console.log(bar_data);

        var bar_x = d3.scaleLinear().domain(d3.extent(bar_data)).range([0,width]).nice();
        var bar_y = d3.scaleBand().domain(bar_data.map(function(d,i){return i;})).range([0, height]);
        var bar_xAxis =d3.axisBottom(bar_x).ticks(7);
        var bar_gx = g.append("g")
                      .attr("class","bar_axis bar_axis--x")
                      .attr("transform", "translate(0," + height + ")")
                      .style('stroke','1px')
                      .call(bar_xAxis);

        var bar_yAxis = d3.axisLeft(bar_y).tickSize(0).tickFormat("");

        var bar_gx = g.append("g")
                      .attr("class","bar_axis bar_axis--y")
                      .attr("transform", "translate(" + bar_x(0) + ",0)")
                      .style('stroke','1px')
                      .call(bar_yAxis);

          // text label for the y axis
         g.append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 0 - margin.left)
          .attr("x",0 - (height / 2))
          .attr("dy", "1em")
          .style("text-anchor", "middle")
          .style("fill","black")
          .text(data_id);


        var bars =  g.selectAll(".bar")
            .data(bar_data)
            .enter().append("rect")
            .attr("class","bar");


      bars.attr("x", function(d) { return bar_x(Math.min(0, d)); })
          .attr("y", function(d,i) { return bar_y(i); })
          .attr("width", function(d) { return Math.abs(bar_x(d) - bar_x(0)); })
          .attr("height", bar_y.bandwidth())
          .style("fill",function(d){return (d<0 ? "darkorange":"steelblue");});

  }
}



function DrawHiddenLayer(selectedDot)
{

  $("#ImageDiv").empty();

  var svgWidth = document.getElementById("ImageDiv").offsetWidth;
  var svgHeight = document.getElementById("ImageDiv").offsetHeight;
  var image_svgWidth = document.getElementById("ImageDiv").offsetWidth;
  var image_svgHeight = document.getElementById("ImageDiv").offsetHeight;
  var margin = {top:0, bottom:0, left:0, right:0};
  var width = +svgWidth- margin.left-margin.right;
  var height = +svgHeight -margin.top- margin.bottom;

  var svg = d3.select("#ImageDiv")
              .append("svg")
              .attr("id","ImageSvg")
              .attr("width", svgWidth)
              .attr("height", svgHeight)
              .append("g")
              .attr("transform","translate("+margin.left+","+margin.top+")");


  var image_svg = svg.append("svg")
                        .classed("image",true)
                        .attr("width",image_svgWidth)
                        .attr("height",image_svgHeight);


  var images = image_svg.selectAll("image").data([0]).enter().append("svg:image")
                        .attr("xlink:href","static/data/mnist/test-images/"+selectedDot+".png")
                        .attr("width",image_svgWidth)
                        .attr("height",image_svgWidth);

  d3.request("http://0.0.0.0:5000/instance_data")
          .header("X-Requested-With", "XMLHttpRequest")
          .header("Content-Type", "application/x-www-form-urlencoded")
          .post(JSON.stringify([nn_chosen,epoch_chosen,class_chosen,selectedDot]), function(e)
            {
              query_instance_data = JSON.parse(e.response);
              DrawBarCharts(query_instance_data);
            });
}


function DrawScatterPlot(query_data)
{
  $("#SPReset").show();

  var correctness = query_data["correctness"];
  var data_points = query_data["position"];
  var indices = query_data["index"];

  d3.select("#ScatterPlotDiv").selectAll("svg").remove();
  d3.select("#ScatterPlotDiv").selectAll("text").remove();

  var selectedDot = null;

  var svgWidth = 450;
  var svgHeight = 450;
  var margin = {top:50, bottom:50, left:50, right:50};
  var width = +svgWidth- margin.left-margin.right;
  var height = +svgHeight -margin.top- margin.bottom;

  var color = d3.scaleOrdinal(d3.schemeCategory10);


  var x = d3.scaleLinear().range([0,width]).nice();
  var y  = d3.scaleLinear().range([height,0]).nice();
  var xMin= d3.extent(data_points.map(function(d){return d[0];}))[0] *1.05;
  var xMax= d3.extent(data_points.map(function(d){return d[0];}))[1] *1.05;
  var yMin = d3.extent(data_points.map(function(d){return d[1];}))[0] * 1.05;
  var yMax = d3.extent(data_points.map(function(d){return d[1];}))[1] * 1.05;

  x.domain([xMin,xMax]);
  y.domain([yMin,yMax]);

  var xAxis = d3.axisBottom(x).tickSize(-height);
  var yAxis = d3.axisLeft(y).tickSize(-width);

  d3.select('#ScatterPlotDiv').append("text")
    .attr("id","scatterplot_title")
    .style("text-anchor", "middle")
    .style("fill","black")
    .text("NN: "+nn_chosen+" Epoch: "+ epoch_chosen + " Class: "+ classes[class_chosen] +" Accuracy_Change: " + data_chosen[epoch_chosen-1][classes[class_chosen]].toFixed(4));


  //Zoom function
  var  zoomBeh = d3.zoom()
        .scaleExtent([1,10])
        .on("zoom", zoom);

    var svg = d3.select('#ScatterPlotDiv')
      .append('svg')
      .attr("id",'ScatterPlot')
      .attr("width", svgWidth)
      .attr("height", svgHeight)
      .append("g")
      .attr("transform","translate("+margin.left+","+margin.top+")")
      .call(zoomBeh);



    var gX = svg.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .style('stroke','1px')
        .call(xAxis);

    var gY = svg.append("g")
        .attr("class", "axis axis--y")
        .style('stroke','1px')
        .call(yAxis);

    var objects = svg.append("svg")
        .classed("objects", true)
        .attr("width", width)
        .attr("height", height);

        objects.append("svg:line")
        .classed("axisLine hAxisLine", true)
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", width)
        .attr("y2", 0)
        .attr("transform", "translate(0," + height + ")");

        objects.append("svg:line")
        .classed("axisLine vAxisLine", true)
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", 0)
        .attr("y2", height);

        objects.selectAll(".dot")
        .data(data_points)
        .enter().append("circle")
        .classed("dot", true)
        .attr("r", 3)
        .attr("cx", function(d){return x(d[0])})
        .attr("cy",function(d){return y(d[1])})
        .style("fill",function(d,i){
          if(correctness[i]){
            return "steelblue";
          }
          else{
            return "Red";
          }
         })
        .on("mouseover",function(d){
            d3.select(this).style("fill","orange");
        })
        .on("mouseout",function(d,i){
          if(indices[i]!=selectedDot){
            d3.select(this).style("fill", function(){
              if(correctness[i]){
                return "steelblue";
              }
              else{
                return "Red";
              }
            });
          }
        })
        .on("click", function(d,i){
          if(selectedDot == null){
            selectedDot = indices[i];
            d3.select(this).style("fill","orange");
            console.log("Image Index: ",selectedDot);
            $("##HiddenLayerDiv").empty();
            DrawHiddenLayer(selectedDot);
          }
          else if(selectedDot != indices[i]){
            selectedDot =indices[i];
            d3.selectAll(".dot").style("fill",function(d,i){
              if(correctness[i]){return "steelblue";}
              else{return "Red";}
            });
            d3.select(this).style("fill","orange");

            console.log("Image Index: ",selectedDot);
            $("#HiddenLayerDiv").empty();
            DrawHiddenLayer(selectedDot);
          }

        });

      function zoom() {
      //console.log(d3.event.transform.k +"  "+ d3.event.transform.x+ "  "+ d3.event
       // .transform.y);
        if (d3.event.transform.k === 1) {d3.event.transform.y = 0; d3.event.transform.x =0;}
        var new_xScale = d3.event.transform.rescaleX(x);
        var new_yScale = d3.event.transform.rescaleY(y);
      // update axes
    gX.call(xAxis.scale(new_xScale));
    gY.call(yAxis.scale(new_yScale));

      svg.selectAll(".dot")
          .attr("transform", d3.event.transform);
    }

        d3.select("#SPReset").on("click", reSet);

        function reSet() {
          svg.transition()
          .duration(750)
          .call(zoomBeh.transform, d3.zoomIdentity);

        }
}

function DrawStreamGraph(data_chosen_pre,data_chosen)
{

    //console.log(d3.event.transform);



    svg.selectAll(".axis").remove();


    var stack = d3.stack().keys(classes)
        .order(d3.stackOrderNone)
        .offset(d3.stackOffsetWiggle);

    var series = stack(data_chosen);

    if(x === undefined){
        //console.log('here new x');
        x = d3.scaleLinear()
        .domain([0, num_of_epoch])
        .range([0, width]);
    }
    new_x = x;
    y = d3.scaleLinear()
        .domain([d3.min(series, stackMin  ), d3.max(series, stackMax)])
        .range([height, 0]);

    function stackMax(layer) {
        return d3.max(layer, function(d) { return d[1]; });
    }

    function stackMin(layer) {
      return d3.min(layer, function(d) { return d[0]; });
    }

    var area = d3.area()
            .x(function(d, i) { return x(i+1); })
            .y0(function(d) { return y(d[0]); })
            .y1(function(d) { return y(d[1]); })
            .curve(d3.curveBasis);

    var xAxis = d3.axisBottom(x);
    var gX = svg.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .style('stroke','1px')
        .call(xAxis);

    var yAxis = d3.axisLeft(y);
    var gY = svg.append("g")
        .attr("class", "axis axis--y")
        .style('stroke','1px')
        .call(yAxis);

    function zoomed() {
        // console.log(d3.event.transform);
        //console.log
        new_x = d3.event.transform.rescaleX(x);
        gX.call(xAxis.scale(new_x));
        svg.selectAll(".area").attr("d", area.x(function(d,i) { return new_x(i+1); }));
        // svg.attr('transform', d3.event.transform);
    }

    zoom  = d3.zoom()
        .scaleExtent([1,10])
        .on("zoom",zoomed);

    d3.select(svg.node().parentElement).call(zoom);


    if (data_chosen_pre === undefined){

        svg.selectAll(".area")
            .data(series)
            .enter().append("path")
            .attr("class","area")
            .attr("d", area)
            .attr("fill", function(d) { return color(d.key); });

    }else{
        var t;
        var layer0  = stack(data_chosen_pre);
        var layer1 = stack(data_chosen);
        d3.selectAll(".area")
            .data(layer1)
          // .data((t = layer1, layer1 = layer0, layer0 = t))
          .transition()
          .duration(2500)
          .attr("class","area")
          .attr("d", area);
        setTimeout(()=>{zoom.transform(d3.select(svg.node().parentElement), d3.zoomIdentity.scale(1) )}, 2500);

    }


    svg.selectAll(".area")
    .attr("fill", function(d) { return color(d.key); })
    .on('mousemove', function(d,i){

        d3.select(this).style('fill',d3.rgb(color(d.key)).brighter());
        mousex = d3.mouse(this)[0];
        invertedx = Math.round(new_x.invert(mousex));
        //console.log(x.invert(mousex), invertedx);
        tip.show(d.key,data_chosen[invertedx-1]);
        d3.select('.d3-tip').style('left', mousex+100+'px').style('top','100px');
       })
    .on('mouseout', function(d){
        d3.select(this).style('fill', color(d.key));
        tip.hide();
      })
    .on('click',function(d){
        console.log(d.key, invertedx);
        class_chosen=d.key;
        epoch_chosen = invertedx;
        d3.request("http://0.0.0.0:5000/data")
          .header("X-Requested-With", "XMLHttpRequest")
          .header("Content-Type", "application/x-www-form-urlencoded")
          .post(JSON.stringify([nn_chosen,d.key,invertedx]), function(e)
            {
                query_data = JSON.parse(e.response);
                console.log(query_data);

                DrawScatterPlot(query_data);

                // if(onfirstbarchart){
                //     console.log("first");
                //     onfirstbarchart = false;
                //     DrawBarCharts(query_data,'barchart');
                // }
                // else{
                //     console.log("second");
                //     DrawBarCharts(query_data,'barchart2');
                // }
            });
        });

}




function changenum(){
    var e = document.getElementById("selectNumber");
    nn_chosen = e.options[e.selectedIndex].value;
    console.log(nn_chosen)
    data_chosen_pre = data_chosen;
    data_chosen =  data[nn_chosen]['data'];
    DrawStreamGraph(data_chosen_pre,data_chosen);
    //onfirstbarchart=true;
}



function createSvg(sel)
{
    return sel
    .append("svg")
    .attr("id", "streamgraph")
    .attr("width", svgWidth)
    .attr("height", svgHeight);
}

function change_view(){
    $("#selectNumber").val(0).change();
    d3.select('#barchart').selectAll("svg").remove();
    d3.select('#barchart2').selectAll("svg").remove();
    d3.select('#barchart').selectAll("text").remove();
    d3.select('#barchart2').selectAll("text").remove();
    onfirstbarchart=true;

}



$(document).ready(function(){
    DrawStreamGraph(data_chosen_pre,data_chosen);
});



