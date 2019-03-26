
var color= d3.scaleOrdinal(d3.schemeCategory10);
let scales_stackedbar = {};
var index_new = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9];
// var isBrushing = false;
var allow_click= false;
var dot_clicked;
var isBrushing = false;
var scales_hiddenLayerbar={};
var layers=["input","c1","c2","f1","o"];
var doubleClicked=[[null,null],[null,null]];
var legend_clicked = {};
var layer_clicked=null;
var scales_query_scatter={};
var selectedDot_Query = null;
var label_clicked = {};



function IsDicZeros(a){
    for (var i=0;i<classes_n.length;i++){
        if(a[classes_n[i]] != 0) return false;
    }
    return true;
}


function InitialDic(data,len){
    for(var i =0;i<len;i++){
        data[i] = 0;
    }
}


for(var i =0;i<classes_n.length;i++){
    legend_clicked[classes_n[i]] = 0;
}

d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};


function ImageHighlight(){
    if(IsDicZeros(legend_clicked)){
        d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",false);
    }
    else{
    d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return legend_clicked[classes[tsne_data[k].label]]==1?false:true});
    }
}


function createStreamGraph(){


    var div_height = document.getElementById("streamgraphdiv").offsetHeight;
    var div_width =  document.getElementById("streamgraphdiv").offsetWidth;


    var svgWidth = div_width;
    var svgHeight= div_height / num_of_nn.length ;


    for (var i=0; i < num_of_nn.length; i++){
        var modelID = "model"+ (num_of_nn[i]).toString();

        d3.select("#streamgraphdiv")
          .append("svg")
          .attr("id",modelID)
          .attr("width", svgWidth)
          .attr("height", svgHeight);

        //DrawStreamGraph(accuracy_data[i].data,modelID);
       DrawStackedBarChat(accuracy_data[i].data,modelID);

    }


}

function DrawLegend(){
    var classes = classes_n;
    var div_height = document.getElementById("legend_div").offsetHeight;
    var div_width = document.getElementById("legend_div").offsetWidth;

    d3.select('#legend_div')
      .append("svg")
      .attr("id","SvgLegend")
      .attr("width",div_width)
      .attr("height",div_height);

    var margin = {top:5, bottom:5, left:5, right:5};
    var width = +div_width- margin.left-margin.right;
    var height = +div_height -margin.top- margin.bottom;
    var offset_x =width/5;
    var offset_y =height/3+5;
    //console.log(legend_clicked);
    var svg = d3.select("#SvgLegend")
                .append("g")
                .attr("transform","translate("+margin.left+","+margin.top+")");



    var legend = svg.selectAll(".legend")
                    .data(classes)
                    .enter()
                    .append("g")
                    .attr("class", "legend")
                    .attr("transform",function(d,i){
                        translate_x = (i%5)*offset_x;
                        translate_y = parseInt(i/5)*offset_y;
                        return "translate("+translate_x+","+translate_y+")";
                    });



    legend.append('rect')
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 10)
        .attr("height", 10)
        .style("fill", function (d, i) {
            return color(i);
        })
        .on("mouseover",function(d,i){
            d3.select(this).style("cursor", "pointer");
                legend.select("rect").style("opacity",function(k,j){return legend_clicked[classes[j]]==1||i==j ? 1:0.2});

            if(isBrushing==false){
                d3.selectAll(".bar").style("opacity",function(k,j){return legend_clicked[classes[index_new[k.index]]]==1||i==index_new[k.index]? 1:0.2});
                d3.selectAll(".dot").classed("unSelectedDot",function(k,j){return legend_clicked[classes[k.label]]==1||i==k.label?false:true;});
            }
            else{
                d3.selectAll(".dot.brushed").classed("unSelectedDot",function(k,j){return legend_clicked[classes[k.label]]==1||i==k.label?false:true});
                d3.selectAll(".sub_bar").selectAll(".instances.bar").style("opacity",function(k,j){return legend_clicked[classes[index_new[k.index]]]==1||i==index_new[k.index]? 1:0.2});
                d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return legend_clicked[classes[tsne_data[k].label]]==1||tsne_data[k].label==i?false:true});


            }
        })
        .on("mouseout",function(){
            if(IsDicZeros(legend_clicked)){
                //gray out all the others
                if(isBrushing==false){
                    legend.selectAll("rect").style("opacity",1);
                    d3.selectAll(".bar").style("opacity",1);
                    d3.selectAll(".dot").classed("unSelectedDot",false);
                }
                else{
                    legend.selectAll("rect").style("opacity",1);
                    d3.selectAll(".dot").classed("unSelectedDot",false);
                    d3.selectAll(".sub_bar").selectAll(".instances.bar").style("opacity",1);

                }
                d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",false);
            }
            else{
                if(isBrushing==false){
                    d3.select(this).style("opacity",function(k){return legend_clicked[k]==1?1:0.2});
                    d3.selectAll(".bar").style("opacity",function(k,j){return legend_clicked[classes[index_new[k.index]]]==1? 1:0.2});
                    d3.selectAll(".dot").classed("unSelectedDot",function(k,j){return legend_clicked[classes[k.label]]==1?false:true});
                }
                else{
                    d3.select(this).style("opacity",function(k){return legend_clicked[k]==1?1:0.2});
                    d3.selectAll(".dot").classed("unSelectedDot",function(k,j){return legend_clicked[classes[k.label]]==1?false:true});
                    d3.selectAll(".sub_bar").selectAll(".instances.bar").style("opacity",function(k,j){return legend_clicked[classes[index_new[k.index]]]==1? 1:0.2});
                     d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return legend_clicked[classes[tsne_data[k].label]]==1?false:true});
                }
            }
        })
        .on("click",function(d,i){
            if(legend_clicked[d] === 1){
                legend_clicked[d] = 0;
                d3.select(this).classed("selected_legend",false).style('opacity',0.2);
            }
            else{
                legend_clicked[d] = 1;
                d3.select(this).classed("selected_legend",true).style('opcacity',1);
            }

        });

    legend.append('text')
           .attr("x", 12)
           .attr("y", 10)
           .text(function (d, i) {return d})
           .attr("class", "textselected")
           .style("text-anchor", "start")
           .style("font-size", 15)
           .style("fill","black");

}


 function DrawStackedBarChat(data,modelID){
    var margin = {top:10, bottom:20, left:25, right:10};
    var bBox = document.getElementById(modelID).getBoundingClientRect();
    var svgWidth = bBox.width;
    var svgHeight=  bBox.height;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    var clicked = false;

    var tooltip = d3.select('#streamgraphdiv').append('div').attr('class', 'hidden tooltip '+modelID);

    function make_button(c_focus,epoch,c1,value1,c2,value2){
        var button;
        if(c_focus!=c1 && c_focus!=c2){
            button = `<tr><td><button type='button' class="classes_button" epoch = '${epoch}' c1='${c1}' c2='${c2}' modelid = '${modelID}'> ${c1} :${value1.toFixed(4)}</br> ${c2} :${value2.toFixed(4)} </button></td></tr>`;
        }

        else if (c_focus==c1){
            button = `<tr><td><button type='button' class="classes_button" epoch = '${epoch}' c1='${c1}' c2='${c2}' modelid = '${modelID}'> <font color="red"> ${c1} :${value1.toFixed(4)} </font></br> ${c2} :${value2.toFixed(4)} </button></td></tr>`;

        }

        else{
            button = `<tr><td><button type='button' class="classes_button" epoch = '${epoch}' c1='${c1}' c2='${c2}' modelid = '${modelID}'> ${c1} :${value1.toFixed(4)} </br><font color="red"> ${c2} :${value2.toFixed(4)} </font> </button></td></tr>`;

        }
        return button;
    }

    function tip_show(c_focus,data){
       // console.log(d,i);
        var e = data.epoch;
        // var k = d;
        var table = `<div>EPOCH: ${e} <button type ='button' class= "${modelID} x"> X </button>`;
        table += `<table class="tip_table">`;
        for(var i=0 ; i<10; i++){
            table+= make_button(c_focus,e,classes[i],data[classes[i]],classes[i+10],data[classes[i+10]]);
            //table+= make_button(1,c_focus,e,classes[i+10],data[classes[i+10]]);
        }
        table+=`</table></div>`;
        //console.log(table);
        return table;
    }


    d3.select("#"+modelID).append("defs").append("clipPath")
      .attr("id","clip")
      .append("rect")
      .attr("width",width)
      .attr("height",height);

    var svg = d3.select("#"+modelID).append("g")
                .attr("class","inner_space")
                .attr("transform","translate("+margin.left+","+margin.top+")");



    //console.log(data);
    var stack = d3.stack()
                   .keys(classes)
                   .offset(d3.stackOffsetDiverging);

    var series = stack(data);


    var x  = d3.scaleLinear()
               .domain([0,num_of_epoch+1])
               .range([width/num_of_epoch/2, width- width/num_of_epoch/2]);

    var new_x = x;


    var y = d3.scaleLinear()
        .domain([d3.min(series, stackMin ), d3.max(series, stackMax)])
        .rangeRound([height, 0]);

    scales_stackedbar[modelID] = {};
    scales_stackedbar[modelID].x = x;
    scales_stackedbar[modelID].y = y;

    function stackMax(layer) {
        return d3.max(layer, function(d) { return d[1]; });
    }

    function stackMin(layer) {
      return d3.min(layer, function(d) { return d[0]; });
    }

    // console.log(series);
    for(row of series){
        row = row.map(d=>{
            d.index=row.index;
            return d;
        });
    }
   //console.log(series);


    svg.append("g")
       .selectAll("g")
       .data(series)
       .enter().append("g")
       .attr("class","group_class")
       .attr("fill", function(d,i){return color(index_new[d.index]);})
       .selectAll("rect")
       .data(function(d){return d;})
       .enter().append("rect")
       .attr('class','bar')
       .attr("width", (width / num_of_epoch) - 1)
       .attr("x", function(d){return (x(d.data.epoch) - width / num_of_epoch / 2) + 1;})
       .attr("y", function(d) { return y(d[1]); })
       .attr("height", function(d) { return y(d[0]) - y(d[1]); });

    var xAxis = d3.axisBottom(x);

    var gX = svg.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + y(0) + ")")
        .style('stroke','1.0 px')
        .call(xAxis);

    var yAxis = d3.axisLeft(y);
    var gY = svg.append("g")
        .attr("class", "axis axis--y")
        .style('stroke','1.0 px')
        .style("font","7px sans-serif")
        .call(yAxis);

    function zoomed() {
        // console.log(d3.event.transform);
        //console.log
        if (d3.event.transform.k === 1) {d3.event.transform.y = 0; d3.event.transform.x =0;}
        new_x = d3.event.transform.rescaleX(x);
        scales_stackedbar[modelID].x = new_x;

        gX.call(xAxis.scale(new_x));
        var num_epoch_ = parseInt(new_x.invert(width-width/num_of_epoch/2) - new_x.invert(width/num_of_epoch/2));
         svg.selectAll(".bar")
           .attr('width',((width / num_epoch_) - 1))
           .attr("x",function(d,i){return (new_x(d.data.epoch)- width/num_epoch_ / 2)+1;});
        // svg.attr('transform', d3.event.transform);
    }

    zoom  = d3.zoom()
        .scaleExtent([1,10])
        .on("zoom",zoomed);
    d3.select(svg.node().parentElement).call(zoom);


 svg.selectAll(".bar")
    .on('mouseover', function(d,i){
        //console.log(d);
        if(clicked==false && isBrushing==false){
            d3.select(this).style('fill',d3.rgb(color(index_new[d.index])).brighter());
            d3.select(this).style("cursor", "pointer");
            mousex = d3.mouse(this)[0];
            //invertedx = Math.round(scales_stackedbar[modelID].x.invert(mousex));
            invertedx = Math.round(new_x.invert(mousex));
            //console.log(data[invertedx-1]);
            tooltip.classed("hidden",false)
                   .attr('style', 'left:' + (d3.event.pageX -320) + 'px; top:' + (d3.event.pageY - 100) + 'px')
                   .html(tip_show(classes[d.index],data[invertedx-1]));
        }
        $("."+modelID+".x").click(function(){
            tooltip.classed('hidden',true);
            clicked=false;
            // d3.selectAll(".dot").style("fill",function(d){return color(d.label);}).classed("unSelectedDot",false);

        });

        $(".classes_button").click(function(){
            d3.request("http://0.0.0.0:5000/class_data")
              .header("X-Requested-With", "XMLHttpRequest")
              .header("Content-Type", "application/x-www-form-urlencoded")
              .post(JSON.stringify([$(this).attr('epoch'),$(this).attr('c1'),$(this).attr('modelid')]), function(e)
                {
                    var query_data = JSON.parse(e.response);
                    console.log(query_data);
                    UpdateScatterPlot(query_data);

                    //DrawScatterPlot(query_data);

                });
        });
       })
    .on('mouseout', function(d){
        if(isBrushing==false){
            d3.select(this).style('fill', color(index_new[d.index]));
            d3.select(this).style("cursor", "default");
            //console.log(clicked);
            if(clicked==false){
            tooltip.classed('hidden', true);
            }
          }
    })
    .on('click',function(){if(isBrushing==false){
        clicked=true;
        console.log("clicked");

    }});

 }


function ShowImage(idx_list){

    var div = d3.select("#ImagesDiv");
    var svgHeight = document.getElementById("ImagesDiv").offsetHeight;
    var svgWidth =  document.getElementById("ImagesDiv").offsetWidth;
    var margin = {top:10, bottom:10, left:10, right:10};
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    $("#ImagesDiv").empty();


    if(idx_list[0]!=null){
        //console.log(idx_list);

        var row = Math.ceil(Math.sqrt(idx_list.length));


        var y = d3.scaleBand().domain(idx_list.map(function(d,i){return Math.floor(i/row);})).range([0,height]);

        var  x = d3.scaleBand().domain(idx_list.map(function(d,i){return i%row;})).range([0,width]);

        var svg = div.append('svg').attr("id","ImageSvg").attr("width", svgWidth).attr("height",svgHeight).append('g').attr("transform","translate("+margin.left+","+margin.top+")");

        //console.log(idx_list);

        svg.selectAll(".image").data(idx_list).enter().append("svg:image").attr("class","image")
            .attr("xlink:href",function(d,i){return "static/data/fashion-mnist/test-images/"+d+".png"})
            .attr("width",width/row)
            .attr("height",height/row)
            .attr("x",function(d,i){return x(i%row);})
            .attr("y", function(d,i){return y(Math.floor(i/row));})
            .on("click",function(d,i){
                if(dot_clicked==null || dot_clicked!=d){
                    dot_clicked = d;
                    svg.selectAll(".image").classed("unselectedImage",true);
                    d3.select(this).classed("unselectedImage",false);
                    d3.selectAll(".dot").classed("selectedDot",function(k,j){return (k.index==d)?true:false});
                    d3.selectAll(".selectedDot").moveToFront();
                   // DrawSlider(dot_clicked);
                }
                else{
                    if(isBrushing==false){
                        dot_clicked=null;
                        svg.selectAll(".image").classed("unselectedImage",false);
                        d3.selectAll(".dot").classed("selectedDot",false);
                    }
                    else{
                        dot_clicked=null;
                        d3.selectAll(".dot").classed("selectedDot",false);
                        ImageHighlight();
                    }

                }
                DrawSliders(dot_clicked);
            });
    }

}


function argmaxDiff(data){
    var max_value = 0;
    var max_index = 0;
    for(var i=1;i<data.length;i++){
        var diff = data[i]-data[i-1];
        if(Math.abs(diff)>max_value){
            max_value = Math.abs(diff);
            max_index = i;
        }
    }
    console.log(max_value,max_index);
    return max_index;

}

function SubmitInstanceData(dot_clicked){

    if(doubleClicked[0][0]!=null && doubleClicked[0][1]!=null&& doubleClicked[1][0]!=null && doubleClicked[1][1]!=null){
        if (dot_clicked==null){return}

        d3.request("http://0.0.0.0:5000/instance_data")
                  .header("X-Requested-With", "XMLHttpRequest")
                  .header("Content-Type", "application/x-www-form-urlencoded")
                  .post(JSON.stringify([doubleClicked[0][0],doubleClicked[0][1],doubleClicked[1][0],doubleClicked[1][1],dot_clicked]), function(e)
                    {
                        var query_data = JSON.parse(e.response);
                        console.log(query_data);
                        DrawHiddenLayerDiv(query_data,dot_clicked);
                    });

    }

}

function DrawSliders(dot_clicked){

    var div = d3.select("#slider-epoch");
    var svgHeight = document.getElementById("slider-epoch").offsetHeight/2;
    var svgWidth =  document.getElementById("slider-epoch").offsetWidth;
    var margin = {top:10, bottom:20, left:30, right:10};
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    $("#slider-epoch").empty();

    if(dot_clicked==null){return}

    var data = loss_diff_data[dot_clicked];

    var x = d3.scaleLinear().domain([0, 100]).range([0, width]);

    for (var i =0;i<num_of_nn.length;i++){

        DrawSlider(i);
    }

    SubmitInstanceData(dot_clicked);

    function DrawSlider(i){
        var svg  = div.append("svg").attr("width",svgWidth).attr("height",svgHeight).attr("id","loss_bar_model"+num_of_nn[i].toString()).append("g").attr("transform","translate("+margin.left+","+margin.top+")");

        var default_index_min = 0;

        var default_index_max = 100;

        doubleClicked[i][1] = default_index_max;
        doubleClicked[i][0] = default_index_min;


        svg.append("text")
            .classed("text_loss",true)
            .attr("x",width-220)
            .attr("y",2)
            .style("fill","black")
            .text("Epoch: "+(default_index_min).toString()+" ~ " +default_index_max.toString()+" Loss_Change: "+(data[i][default_index_max]- data[i][default_index_min]).toFixed(4));



        var y = d3.scaleLinear().domain(d3.extent(data[i])).range([height,0]);

        // 7. d3's line generator
        var line = d3.line()
            .x(function(d, j) { return x(j); }) // set the x values for the line generator
            .y(function(d) { return y(d); }) // set the y values for the line generator
            .curve(d3.curveMonotoneX); // apply smoothing to the line

        var yAxis = d3.axisLeft(y).ticks(3).tickSize(2);

        var xAxis = d3.axisBottom(x).ticks(20).tickSize(1);

        // 3. Call the x axis in a group tag
        svg.append("g")
            .attr("class", "x line_axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis); // Create an axis component with d3.axisBottom


        // 4. Call the y axis in a group tag
        svg.append("g")
            .attr("class", "y line_axis")
            .call(yAxis); // Create an axis component with d3.axisLeft

        // 9. Append the path, bind the data, and call the line generator
        svg.append("path")
            .datum(data[i]) // 10. Binds data to the line
            .attr("class", "loss_line") // Assign a class for styling
            .attr("d", line);

        // 12. Appends a circle for each datapoint
        svg.selectAll(".loss_dot")
            .data(data[i])
            .enter().append("circle") // Uses the enter().append() method
            .attr("class", "loss_dot") // Assign a class for styling
            .attr("cx", function(d, j) { return x(j) })
            .attr("cy", function(d) { return y(d) })
            .attr("r", 3)
            .style("fill",function(d,j){console.log(predict_data[dot_clicked][i][j]);return color(predict_data[dot_clicked][i][j]);})
            .style("stroke","#fff")
            .style("stroke-width",0.2);

        var brush = d3.brushX().extent([[0,0],[width,height]]).on("end",brushed);


        function brushed(){
            if (!d3.event.sourceEvent) return; // Only transition after input.
            if (!d3.event.selection) return; // Ignore empty selections.
            var d0 = d3.event.selection.map(x.invert),
                d1 = d0.map(Math.round);
            d3.select(this).transition().call(d3.event.target.move, d1.map(x));

            if(d1[1]>d1[0]){

                console.log(d1);
                doubleClicked[i][0] = d1[0];
                doubleClicked[i][1] = d1[1];
                SubmitInstanceData(dot_clicked);
                svg.selectAll(".text_loss").remove();

                svg.append("text")
                    .classed("text_loss",true)
                    .attr("x",width-220)
                    .attr("y",2)
                    .style("fill","black")
                    .text("Epoch: "+(doubleClicked[i][0]).toString()+" ~ " +(doubleClicked[i][1]).toString()+" Loss_Change: "+(data[i][doubleClicked[i][1]]- data[i][doubleClicked[i][0]]).toFixed(4));
            }
        }


        svg.append("g").attr("class","brush").call(brush);
        // .call(brush.move,[default_index_min,default_index_max].map(x));

    }

}




function UpdateScatterPlot(query_data){
    var class_indices = query_data['whole_index'];
    var loss_before = query_data['loss_before'];
    var loss_after = query_data['loss_after'];

    var targets_dots = d3.selectAll(".dot").classed("unSelectedDot",function(k,i){return class_indices.includes(i)?false:true});

    var target_dots = d3.selectAll(".dot:not(.unSelectedDot)");

    var loss_diff = loss_after.map(function(d,i){return d-loss_before[i];});

    var min= d3.extent(loss_diff)[0];
    var max = d3.extent(loss_diff)[1];

    var max_abs = Math.max(Math.abs(min),Math.abs(max));
    var min_abs = max_abs*(-1.0);

    var color_loss = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);


    target_dots.style('fill',function(d,i){return color_loss(loss_diff[i]);});

}

function createScatterPlot(data){

    $("#SPReset").show();

    var svgHeight = document.getElementById("ScatterPlotDiv").offsetHeight;
    var svgWidth =  document.getElementById("ScatterPlotDiv").offsetWidth;
    var margin = {top:30, bottom:50, left:20, right:20};
    var width = +svgWidth- margin.left-margin.right;
    dot_clicked = null;
    var height = +svgHeight -margin.top- margin.bottom;

    var x = d3.scaleLinear().range([0,width]).nice();
    var y  = d3.scaleLinear().range([height,0]).nice();
    var xMin= d3.extent(data.map(function(d){return d.data[0];}))[0] *1.05;
    var xMax= d3.extent(data.map(function(d){return d.data[0];}))[1] *1.05;
    var yMin = d3.extent(data.map(function(d){return d.data[1];}))[0] * 1.05;
    var yMax = d3.extent(data.map(function(d){return d.data[1];}))[1] * 1.05;

    x.domain([xMin,xMax]);
    y.domain([yMin,yMax]);

    var xAxis = d3.axisBottom(x).tickSize(-height);
    var yAxis = d3.axisLeft(y).tickSize(-width);


    var svg = d3.select('#ScatterPlotDiv')
      .append('svg')
      .attr("id",'ScatterPlot')
      .attr("width", svgWidth)
      .attr("height", svgHeight)
      .append("g")
      .attr("transform","translate("+margin.left+","+margin.top+")");



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

        var brush = d3.brush().on("brush", highlightBrushedDots).on("end",filterBarChart);

        objects.append("g").attr("class","brush").call(brush);

        objects.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("r", 3)
        .attr("cx", function(d){return x(d.data[0])})
        .attr("cy",function(d){return y(d.data[1])})
        .style("fill",function(d,i){
          return color(d.label);
        })
        .on("mouseover",function(d,i){
            if(!isBrushing){
                d3.select(this).style("cursor", "pointer");
                d3.select(this).classed("selectedDot",true);
                ShowImage([d.index]);
                d3.select(this).moveToFront();
            }
            else if(!d3.select(this).attr("class").includes("non_brushed")){
                d3.select(this).style("cursor", "pointer");
                d3.select(this).classed("selectedDot",true);
                d3.select(this).moveToFront();
                d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return k==d.index||k==dot_clicked?false:true});

            }
        })
        .on("mouseout",function(d,i){
            if(!isBrushing){
                if(dot_clicked==null || dot_clicked!=d.index){
                    d3.select(this).style("cursor", "default");
                    d3.select(this).classed("selectedDot",false);
                    ShowImage([dot_clicked]);
                }
            }
            else if(!d3.select(this).attr("class").includes("non_brushed")){
                if(dot_clicked==null){
                    d3.select(this).style("cursor", "default");
                    d3.select(this).classed("selectedDot",false);
                    ImageHighlight();
                }
                else if(dot_clicked!=d.index){
                    d3.select(this).style("cursor", "default");
                    d3.select(this).classed("selectedDot",false);
                    d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return k==dot_clicked?false:true});
                }
                else{
                    d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return k==dot_clicked?false:true});
                }
            }
        })
        .on("click",function(d,i){
            if(!d3.select(this).attr("class").includes("non_brushed")){
                if(dot_clicked!=d.index){
                    dot_clicked= d.index;
                    d3.selectAll(".selectedDot").classed("selectedDot",false);
                    d3.select(this).classed("selectedDot",true);
                    d3.select(this).moveToFront();
                    SubmitInstanceData(dot_clicked);
                    DrawSliders(dot_clicked);
                }
                else{dot_clicked=null}
            }
        });


    d3.select("#SPReset").on("click", reSet);


    d3.select("#findDivergingBtn").on("click", showDivergingPoints);

    function reSet() {
          svg.select("g.brush").call(brush.move, null);
          d3.selectAll('rect.instances').remove();
          d3.selectAll('.bar')
            .style("fill", function(d,i){return color(index_new[d.index]);})
            .style("opacity",1);
        d3.selectAll(".legend").selectAll('.rect').style("opacity",1);
        isBrushing= false;
        $("#ImagesDiv").empty();
        isBrushing = false;
        for(var i =0;i<classes_n.length;i++){
            legend_clicked[classes_n[i]] = 0;
        }
        d3.select("#SvgLegend").selectAll("rect").classed('selected_legend',false).style("opacity",1);
          d3.selectAll(".dot").style("fill",function(d){return color(d.label);}).classed("unSelectedDot",false).classed("non_brushed",false).classed("brushed",false);


    }

    function showDivergingPoints(){
        reSet();
        isBrushing = true;
        svg.selectAll(".dot").filter(function (d,i){ return indices.indexOf(d.index) == -1})
            .classed("non_brushed",true);

        ShowImage(indices);


    }

}



function filterBarChart(){
    if (!d3.event.selection) return

    isBrushing=true;

   // var d_brushed = d3.selectAll(".brushed:not(.unSelectedDot)").data();
   var d_brushed = d3.selectAll(".brushed").data();

    var brushed_index = d_brushed.map(d=>d.index);
    console.log(brushed_index);

    ShowImage(brushed_index);

    isBrushing= true;

    if(IsDicZeros(legend_clicked)){
        d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",false);

    }
    else{
        d3.select("#ImagesDiv").selectAll(".image").classed("unselectedImage",function(k){return legend_clicked[classes_n[tsne_data[k].label]]==1?false:true});
    }



    d3.request("http://0.0.0.0:5000/loss_sub_data")
          .header("X-Requested-With", "XMLHttpRequest")
          .header("Content-Type", "application/x-www-form-urlencoded")
          .post(JSON.stringify(brushed_index), function(e)
            {
                var query_data = JSON.parse(e.response);

                //console.log(query_data);

                for (var i=0; i < num_of_nn.length; i++){
                    var modelID = "model"+ (num_of_nn[i]).toString();

                    DrawLossBar(query_data[i].data,modelID);

                }
                console.log(query_data);
            });

}

function highlightBrushedDots(){
    //diable dot_clicked
    d3.selectAll(".selectedDot").classed("selectedDot",false);
    dot_clicked = null;


    d3.selectAll(".dot").classed("non_brushed",true).classed("brushed",false);

    if(d3.event.selection !=null){
        var brush_coords = d3.brushSelection(this);
     // style brushed circles
        d3.selectAll(".dot").filter(function (){

           var cx = d3.select(this).attr("cx"),
               cy = d3.select(this).attr("cy");

           return isBrushed(brush_coords, cx, cy);
        })
        .classed("non_brushed",false).classed("brushed",true);

    d3.selectAll(".unSelectedDot").classed("unSelectedDot",true);
    }
}

function isBrushed(brush_coords, cx, cy) {

    var x0 = brush_coords[0][0],
        x1 = brush_coords[1][0],
        y0 = brush_coords[0][1],
        y1 = brush_coords[1][1];

    return x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
}


function DrawLossBar(data,modelID){
    var margin = {top:10, bottom:20, left:25, right:10};
    var bBox = document.getElementById(modelID).getBoundingClientRect();
    var svgWidth = bBox.width;
    var svgHeight=  bBox.height;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;

    //console.log(data);

    var x = scales_stackedbar[modelID].x;

    var y = scales_stackedbar[modelID].y;

    var stack = d3.stack()
                   .keys(classes)
                   .offset(d3.stackOffsetDiverging);

    var series = stack(data);


    var svg = d3.select("#"+modelID).select(".inner_space");


    // console.log(series);
    for(row of series){
        row = row.map(d=>{
            d.index=row.index;
            return d;
        });
    }
    //console.log(series);
    svg.selectAll(".sub_bar").remove();

    svg.selectAll(".bar").style("fill",function(){return '#808080'}).style('opacity',0.2);

    var num_epoch_ = parseInt(x.invert(width-width/num_of_epoch/2) - x.invert(width/num_of_epoch/2));

    svg.append("g")
        .attr("class","sub_bar")
        .selectAll("g")
        .data(series)
        .enter().append("g")
        .attr("class","group_class")
        .attr("fill", function(d,i){return color(index_new[i]);})
        .selectAll("rect")
        .data(function(d){return d;})
        .enter().append("rect")
        .attr('class','bar instances ')
        .attr('width',((width / num_epoch_) - 1))
        .attr("x",function(d,i){return (x(d.data.epoch)- width/num_epoch_ / 2)+1;})
        .attr("y", function(d) { return y(d[1]); })
        .attr("height", function(d) { return y(d[0]) - y(d[1]); });
}





function DrawHiddenLayerDiv(data,dot_clicked){
    if(layer_clicked==null){
        d3.select("#HiddenLayerDiv").selectAll("div").remove();
    }
    else{
        d3.select("#HiddenLayerDiv").selectAll("div:not(#QueryScatterPlotDiv)").remove();
    }
    var div_height = document.getElementById("HiddenLayerDiv").offsetHeight;
    var div_width =  document.getElementById("HiddenLayerDiv").offsetWidth;

    var layerHeight = 20;

    var layers=["input","c1","c2","f1","o"];


    var svgWidth = div_width;
    var svgHeight= (div_height-layerHeight) / num_of_nn.length ;


    d3.select("#HiddenLayerDiv")
        .append("div")
        .attr("id","layers")
        .style("width",svgWidth+'px')
        .style("height", layerHeight+'px');

    var label_svg = d3.select("#layers")
                      .append("svg")
                      .attr("class","svglabel")
                      .attr("width",svgWidth)
                      .attr("height",layerHeight);


    label_svg.selectAll(".label").data(layers).enter().append("text")
            .attr("class","layer")
            .attr("id",function(d){return d;})
            .attr("y", 0)
            .attr("x",function(d,i){return i*svgWidth/5+svgWidth/10;})
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .style("fill","black")
            .text(function(d){return d;})
            .on("click",function(d,i){
                SwitchView(d,i);
            });


    function SwitchView(d,i){
        if(layer_clicked==null){
            d3.select("#HiddenLayerDiv").selectAll("svg:not(."+d+")").classed("hidden",true);
            d3.select("#HiddenLayerDiv").selectAll(".image").classed("hidden",false);
            d3.select("#HiddenLayerDiv").selectAll(".svglabel").classed("hidden",false);
            d3.select("#HiddenLayerDiv").selectAll(".layer:not(#"+d+")").classed("hidden",true);
            d3.select("#HiddenLayerDiv").selectAll("#input").classed("hidden",false);
            d3.select("#HiddenLayerDiv").select("#"+d).attr("x",svgWidth/5+svgWidth/10);
            layer_clicked=d;
            DrawSearchHiddenLayer(d,dot_clicked);
        }
        else{
            layer_clicked=null;
            SubmitInstanceData(dot_clicked);
        }
    }


    var input_margin = {top:10, bottom:10, left:10, right:10};

    var inputSvgHeight =div_height- layerHeight;
    var inputSvgWidth = div_width/5;

    var input_width = +inputSvgWidth- input_margin.left-input_margin.right;
    var input_height = +inputSvgHeight -input_margin.top- input_margin.bottom;



    d3.select("#HiddenLayerDiv")
        .append("div")
        .attr("id","input_image")
        .style("float","left")
        .style("width",inputSvgWidth+'px')
        .style("height", inputSvgHeight+'px');

    var image_svg = d3.select('#input_image')
            .append("svg")
            .attr("class","image")
            .attr("id","imageSvg")
            .attr("width", inputSvgWidth)
            .attr("height", inputSvgHeight/2)
            .append("g")
            .attr("id","image_svg")
            .attr("transform","translate("+input_margin.left+","+input_margin.top+")");

    var tooltip = d3.select("#input_image").append('div').attr('class', 'hidden image_tooltip');

    image_svg.selectAll("image").data([0]).enter().append("svg:image")
            .attr("xlink:href","static/data/fashion-mnist/test-images/"+dot_clicked+".png")
            .attr("class","input_image")
            .attr("width",input_height/4-input_margin.top)
            .attr("height",input_height/4-input_margin.top)
            .attr("x",25)
            .attr("y",0)
            .on("mouseover",function(d,i){
                tooltip.classed("hidden",false)
                .style("top","250px")
                .text(classes_n[tsne_data[dot_clicked].label]);
            })
            .on("mouseout",function(d,i){
                tooltip.classed("hidden",true);
            });

    for (var i=0; i < num_of_nn.length; i++){
        var modelID = "hiddenlayer_model"+ (num_of_nn[i]).toString();

        d3.select("#HiddenLayerDiv")
          .append("div")
          .attr("id",modelID)
          .style("width",svgWidth*0.8+'px')
          .style("height", svgHeight+'px')
          .style("float","left");

        d3.select("#input_image")
          .append("svg")
          .attr("id","heatmap"+(i+1).toString())
          .attr("width", inputSvgWidth)
          .attr("height", inputSvgHeight/4)
          .append("g")
          .attr("id","heatmap"+(i+1).toString()+"_g")
          .attr("transform","translate("+(input_margin.left+25)+","+input_margin.top+")");

        //DrawStreamGraph(accuracy_data[i].data,modelID);
       DrawHiddenLayer(data[i],modelID,i+1);

    }
    if(layer_clicked!=null){
        for(var l=0;l<layers.length;l++){
            if(layers[l]!=layer_clicked){
                d3.select("#HiddenLayerDiv").selectAll("svg."+layers[l]).classed("hidden",true);
            }
        }
        d3.select("#HiddenLayerDiv").select("#input_image").selectAll("svg:not(.image)").classed("hidden",true);
        d3.select("#HiddenLayerDiv").selectAll(".svglabel").classed("hidden",false);
        d3.select("#HiddenLayerDiv").selectAll(".layer:not(#"+layer_clicked+")").classed("hidden",true);
        d3.select("#HiddenLayerDiv").selectAll("#input").classed("hidden",false);
        d3.select("#HiddenLayerDiv").select("#"+layer_clicked).attr("x",svgWidth/5+svgWidth/10);
        DrawSearchHiddenLayer(layer_clicked,dot_clicked);
    }


}



function DrawHiddenLayer(data,modelID,index_model){
    //console.log(data);
    InitialDic(label_clicked,classes_n.length);

    var margin = {top:10, bottom:20, left:10, right:10};
    var svgHeight = document.getElementById(modelID).offsetHeight;
    var svgWidth = document.getElementById(modelID).offsetWidth/4;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;

    DrawConvChart(0,12,3,4);
    DrawConvChart(1,4,4,8);
    DrawBarChart(2);
    DrawBarChart(3);
    DrawConvChart(4,28,1,4.64);

    function DrawConvChart(plot_i,rectSize,col_num,gridSize){
        var data_id = data[plot_i].label;
        var data_origin = data[plot_i].data_origin;
        var data_prev = data[plot_i].data_prev;
        var plot_data = data_origin.map(function(d,i){return d - data_prev[i];});

        min = d3.extent(plot_data)[0];
        max = d3.extent(plot_data)[1];

        var max_abs = Math.max(Math.abs(min),Math.abs(max));
        var min_abs = max_abs*(-1.0);

        var color = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);

        scales_hiddenLayerbar[modelID+data_id] = {}
        scales_hiddenLayerbar[modelID+data_id].c = color;


        //var color = d3.scaleLinear().domain(d3.extent(plot_data)).range(["white", "black"]);

        var row_size = rectSize*col_num+1;

        var height_size = plot_data.length/rectSize*col_num+ (plot_data.length/(rectSize*col_num))/rectSize-1;


        // var gridSize = 4;
        //console.log(row_size,height_size,gridSize);

        var svg;

        if(data_id=="input"){
            svg = d3.select('#input_image').select("#heatmap"+index_model.toString()+"_g");
        }

        else{
            svg = d3.select('#'+modelID)
                .append("svg")
                .attr("class",data_id)
                .attr("width", svgWidth)
                .attr("height", svgHeight)
                .append("g")
                .attr("class","inner_space")
                .attr("transform","translate("+margin.left+","+margin.top+")");
        }



        var conv_rects = svg.selectAll("."+data_id)
                            .data(plot_data)
                            .enter().append("rect")
                            .attr("class",data_id);


        conv_rects.attr("x",function(d,i){
                    var index_rect = Math.floor(i/(rectSize**2));
                    var col_in_rect = (i%(rectSize**2))%rectSize;
                    var col_index = index_rect%col_num;
                    return (col_index)*(rectSize+1)*gridSize +col_in_rect*gridSize;
                    })
                  .attr("y",function(d,i){
                    var index_rect = Math.floor(i/(rectSize**2));
                    var row_in_rect = Math.floor((i%(rectSize**2))/rectSize);
                    var row_index = Math.floor(index_rect/col_num);
                    return (row_index)*(rectSize+1)*gridSize + row_in_rect*gridSize;
                    })
                  .attr("width",gridSize)
                  .attr("height",gridSize)
                  .style("fill",function(d,i){return color(d);});

    }


    function DrawBarChart(plot_i)
    {

        var data_id = data[plot_i].label;
        var data_origin = data[plot_i].data_origin;
        var data_prev = data[plot_i].data_prev;
        var plot_data = data_origin.map(function(d,i){return d-data_prev[i];});
        var plot_data_index = Array.apply(null, {length: plot_data.length}).map(Number.call, Number);

        min = d3.extent(plot_data)[0];
        max = d3.extent(plot_data)[1];

        var max_abs = Math.max(Math.abs(min),Math.abs(max));
        var min_abs = max_abs*(-1.0);

        var color = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);

        var svg = d3.select('#'+modelID)
                .append("svg")
                .attr("class",data_id)
                .attr("width", svgWidth)
                .attr("height", svgHeight)
                .append("g")
                .attr("class","inner_space")
                .attr("transform","translate("+margin.left+","+margin.top+")");

        var x = d3.scaleLinear().domain(d3.extent(plot_data)).range([0,width]).nice();
        var y = d3.scaleBand().domain(plot_data.map(function(d,i){return plot_data_index[i];})).range([0, height]);

        scales_hiddenLayerbar[modelID+data_id] = {};
        scales_hiddenLayerbar[modelID+data_id].x = x;
        scales_hiddenLayerbar[modelID+data_id].y = y;
        scales_hiddenLayerbar[modelID+data_id].c = color;


        var xAxis =d3.axisBottom(x).ticks(5);
        var gx = svg.append("g")
                  .attr("class","bar_axis bar_axis--x")
                  .attr("transform", "translate(0," + height + ")")
                  .style('stroke','1px')
                  .call(xAxis);

        var yAxis = d3.axisLeft(y).tickSize(0).tickFormat("");
        //var yAxis = d3.axisLeft(y);

        var gy = svg.append("g")
                  .attr("class","bar_axis bar_axis--y")
                  .attr("transform", "translate(" + x(0) + ",0)")
                  .style('stroke','1px')
                  .call(yAxis);
                  // text label for the y axis



        var bars =  svg.selectAll(".nn_bar")
            .data(plot_data)
            .enter().append("rect")
            .attr("class","nn_bar");


        bars.attr("x", function(d) { return x(Math.min(0, d)); })
          .attr("y", function(d,i) { return y(i); })
          .attr("width", function(d) { return Math.abs(x(d) - x(0)); })
          .attr("height", y.bandwidth())
          .style("fill",function(d){return color(d);});

        //click on the bar to update
        if(data_id=='o'){
            color  = d3.scaleOrdinal(d3.schemeCategory10);
            bars.style("fill",function(d,i){return color(i);});
            bars.on("click",function(d,i){
                console.log(i.toString(),d);
                console.log(label_clicked);
                if(label_clicked[i]==1){
                    label_clicked[i] = 0;
                    d3.select(this).style("stroke","none");
                }
                else{
                    label_clicked[i]=1;
                    d3.select(this).style("stroke","black").style("stroke-width",2);
                }
                UpdateHiddenCharts(modelID,label_clicked,index_model);

            });
        }

    }
}


function UpdateHiddenCharts(modelID,label_clicked,index_model){
    var epoch_chosen;
    if(modelID =="hiddenlayer_model5"){
        epoch_chosen=doubleClicked[0];
    }
    else{
        epoch_chosen=doubleClicked[1];
    }

    if (epoch_chosen[0]==null || epoch_chosen[1]==null  || dot_clicked==null){return}

    d3.request("http://0.0.0.0:5000/grad_instance_data")
          .header("X-Requested-With", "XMLHttpRequest")
          .header("Content-Type", "application/x-www-form-urlencoded")
          .post(JSON.stringify([modelID,epoch_chosen[0],epoch_chosen[1],dot_clicked,label_clicked]), function(e)
            {
                var query_data = JSON.parse(e.response);
                console.log(query_data);
                UpdateConvChart(modelID,query_data[0],index_model);
                UpdateConvChart(modelID,query_data[1],index_model);
                UpdateConvChart(modelID,query_data[2],index_model);
                UpdateBarChart(modelID,query_data[3],index_model);
            });
}


function UpdateConvChart(modelID,data,index_model){
    var margin = {top:10, bottom:20, left:10, right:10};
    var svgHeight = document.getElementById(modelID).offsetHeight;
    var svgWidth = document.getElementById(modelID).offsetWidth/4;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    var data_label = data.label;
    var data_update = data.data;
    var svg;
    if(data_label =="input"){
        //console.log("#heatmap"+modelID[modelID.length-1]+"_g");
        svg = d3.select("#heatmap"+index_model.toString()+"_g");
    }
    else{
        svg = d3.select("#"+modelID).select("."+data_label).select(".inner_space");
    }


    var min = d3.min(data_update);
    var max = d3.max(data_update);

    var max_abs = Math.max(Math.abs(min),Math.abs(max));
    var min_abs = max_abs*(-1.0);



    var color_update = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);

    svg.selectAll("rect."+data_label).data(data_update)
          .transition()
          .duration(2500)
          .style("fill",function(d,i){return color_update(d);});

}

function UpdateBarChart(modelID,data,index_model){
    var margin = {top:10, bottom:20, left:10, right:10};
    var svgHeight = document.getElementById(modelID).offsetHeight;
    var svgWidth = document.getElementById(modelID).offsetWidth/4;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    var data_label = data.label;
    var data_update = data.data;
    var svg;
    var data_index = Array.apply(null, {length: data_update.length}).map(Number.call, Number);
    console.log(data_label);
    if(data_label =="input"){
        //console.log("#heatmap"+modelID[modelID.length-1]+"_g");
        svg = d3.select("#heatmap"+index_model.toString()+"_g");
    }
    else{
        svg = d3.select("#"+modelID).select("."+data_label).select(".inner_space");
    }

    var min = d3.extent(data_update)[0];
    var max = d3.extent(data_update)[1];

    var max_abs = Math.max(Math.abs(min),Math.abs(max));
    var min_abs = max_abs*(-1.0);

    var color_update = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);




    var x_update = d3.scaleLinear().domain(d3.extent(data_update)).range([0,width]).nice();
    var y_update = d3.scaleBand().domain(data_update.map(function(d,i){return data_index[i];})).range([0, height]);

    var xAxis_update =d3.axisBottom(x_update).ticks(5);
    var gx_update =svg.select(".bar_axis--x").call(xAxis_update);
    var yAxis_update = d3.axisLeft(y_update).tickSize(0).tickFormat("");

    var bar_gy_update =svg.select(".bar_axis--y").attr("transform", "translate(" + x_update(0) + ",0)").call(yAxis_update);



    var bars =  svg.selectAll(".nn_bar")
        .data(data_update)
        .transition()
        .duration(2500)
        .attr("x", function(d) { return x_update(Math.min(0, d)); })
        .attr("y", function(d,i) { return y_update(i); })
        .attr("width", function(d) { return Math.abs(x_update(d) - x_update(0)); })
        .attr("height", y_update.bandwidth())
        .style("fill",function(d){return color_update(d);});

}



function DrawSearchHiddenLayer(data_id,dot_clicked){

        var epoch_chosen = doubleClicked[0];
        var epoch2_chosen = doubleClicked[1];

        if (doubleClicked[0][0]==null || doubleClicked[0][1]==null || doubleClicked[1][0]==null || doubleClicked[1][1]==null || dot_clicked==null){return}


        d3.request("http://0.0.0.0:5000/search_instance_data")
        .header("X-Requested-With", "XMLHttpRequest")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .post(JSON.stringify([data_id,doubleClicked,dot_clicked]), function(e)
            {
                var query_data = JSON.parse(e.response);
                //console.log(query_data[0].x);
                if(document.getElementById("QueryScatterPlotDiv") == null){
                    //console.log("DrawLegend")
                    DrawQueryResult(query_data,data_id);
                 }//
                else{
                     UpdateQueryResult(query_data);
                }
            });
}

function UpdateQueryResult(data){

    d3.select("#QueryScatterPlotDiv").selectAll(".hidden").classed("hidden",false);

    var margin = {top:20, bottom:40, left:30, right:20};
    var svgHeight = document.getElementById("HiddenLayerDiv").offsetHeight;
    var svgWidth = 570;
    var width_translate = document.getElementById("HiddenLayerDiv").offsetWidth - svgWidth;
    var width = +svgWidth- margin.left-margin.right;
    var top_translate = document.getElementById("layers").offsetHeight;
    var height = +svgHeight -top_translate-margin.top- margin.bottom;

    var x = d3.scaleLinear().range([0,width]).nice();
    var y  = d3.scaleLinear().range([height,0]).nice();
    var xMin= d3.extent(data.map(function(d){return d.x;}))[0] *0.95;
    var xMax= d3.extent(data.map(function(d){return d.x;}))[1] *1.05;
    var yMin = d3.extent(data.map(function(d){return d.y;}))[0] * 0.95;
    var yMax = d3.extent(data.map(function(d){return d.y;}))[1] * 1.05;

    var min = Math.min(xMin,yMin);
    var max = Math.max(xMax,yMax);
    console.log(min,max);

    x.domain([min-max*0.1,max]);
    y.domain([min-max*0.1,max]);

    var xAxis = d3.axisBottom(x).tickSize(-height);
    var yAxis = d3.axisLeft(y).tickSize(-width);


    var svg = d3.select("#QueryScatterPlotDiv").select(".inner_space");

    var t = d3.transition().duration(1000);


    svg.select(".axis--x").transition(t).call(xAxis);
    svg.select(".axis--y").transition(t).call(yAxis);

    //console.log(data);


    svg.selectAll(".query_dot")
        .data(data, d=>d.index)
        .transition()
        .duration(2000)
        .attr("cx", function(d){return x(d.x)})
        .attr("cy",function(d){return y(d.y)})
        .style("fill",function(d,i){return color(d.label)});


    d3.selectAll(".selectedQueryDot").classed("selectedQueryDot",false);
    d3.select('#HiddenLayerDiv').selectAll(".query_image").remove();
    selectedDot_Query=null;


}

function DrawQueryResult(data,data_id){


    var margin = {top:20, bottom:40, left:30, right:20};
    var svgHeight = document.getElementById("HiddenLayerDiv").offsetHeight;
    var svgWidth = 570;
    var width_translate = document.getElementById("HiddenLayerDiv").offsetWidth - svgWidth;
    var width = +svgWidth- margin.left-margin.right;
    var top_translate = document.getElementById("layers").offsetHeight;
    var height = +svgHeight -top_translate-margin.top- margin.bottom;
    selectedDot_Query = null;
    var new_xScale,new_yScale;

    $("#QueryScatterPlotDiv").remove();

   //Zoom function
    var zoomBeh = d3.zoom()
        .scaleExtent([1,500])
        .on("zoom", zoom);


    d3.select("#HiddenLayerDiv")
      .append("div")
      .attr("id","QueryScatterPlotDiv")
      .style("width",svgWidth+'px')
      .style("height", svgHeight-top_translate+'px')
      .style("position","absolute")
      .style("left",width_translate+'px')
      .style("top",top_translate+'px');


    var svg = d3.select("#QueryScatterPlotDiv")
      .append("svg")
      .attr("class","search")
      .attr("width", svgWidth)
      .attr("height",svgHeight-top_translate)
      .append("g")
      .attr("transform","translate("+margin.left+","+margin.top+")")
      .attr("class","inner_space")
      .call(zoomBeh);


    var x = d3.scaleLinear().range([0,width]).nice();
    var y  = d3.scaleLinear().range([height,0]).nice();
    var xMin= d3.extent(data.map(function(d){return d.x;}))[0] *0.95;
    var xMax= d3.extent(data.map(function(d){return d.x;}))[1] *1.05;
    var yMin = d3.extent(data.map(function(d){return d.y;}))[0] * 0.95;
    var yMax = d3.extent(data.map(function(d){return d.y;}))[1] * 1.05;

    var min = Math.min(xMin,yMin);
    var max = Math.max(xMax,yMax);

    //console.log(min,max);

    x.domain([min-max*0.1,max]);
    y.domain([min-max*0.1,max]);

    var xAxis = d3.axisBottom(x).tickSize(-height);
    var yAxis = d3.axisLeft(y).tickSize(-width);


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


    console.log(data);

    objects.selectAll(".query_dot")
        .data(data,d=>d.index)
        .enter().append("circle")
        .attr("class", "query_dot")
        .attr("r", 3)
        .attr("cx", function(d){return x(d.x)})
        .attr("cy",function(d){return y(d.y)})
        .style("fill",function(d,i){
          return color(d.label);
        })
        .on("mouseover",function(d,i){
            d3.select(this).moveToFront();
            d3.select(this).style("cursor", "pointer");
            d3.select(this).classed("selectedQueryDot",true);
            DrawHiddenLayer_Query(data_id,d);
            DrawImage_Query(d.index);

        })
        .on("mouseout",function(d,i){
            if(selectedDot_Query==null){

                d3.select(this).style("cursor", "default");
                d3.select(this).classed("selectedQueryDot",false);
                DrawHiddenLayer_Query(data_id,data[selectedDot_Query]);
                DrawImage_Query(null);
            }
            else if(selectedDot_Query!=d.index){
                d3.select(this).style("cursor", "default");
                d3.select(this).classed("selectedQueryDot",false);
                DrawHiddenLayer_Query(data_id,data[selectedDot_Query]);
                DrawImage_Query(selectedDot_Query);
            }
        })
        .on("click",function(d,i){
            d3.select(this).moveToFront();
            if(selectedDot_Query!=d.index){
                selectedDot_Query=d.index;
                d3.selectAll(".selectedQueryDot").classed("selectedQueryDot",false);
                d3.select(this).classed("selectedQueryDot",true);
                DrawHiddenLayer_Query(data_id,data[selectedDot_Query]);
                DrawImage_Query(data[selectedDot_Query].index);
            }
            else{selectedDot_Query=null}
        });


    function zoom() {
        if (d3.event.transform.k === 1) {d3.event.transform.y = 0; d3.event.transform.x =0;}
        new_xScale = d3.event.transform.rescaleX(x);
        new_yScale = d3.event.transform.rescaleY(y);
      // update axes
        gX.call(xAxis.scale(new_xScale));
        gY.call(yAxis.scale(new_yScale));

        svg.selectAll(".query_dot")
              .attr("transform", d3.event.transform)
              .attr("r",3/d3.event.transform.k);
    }

}


function DrawImage_Query(selectedDot_index){
    var layerHeight = 20;

    var div_height = document.getElementById("HiddenLayerDiv").offsetHeight;
    var div_width =  document.getElementById("HiddenLayerDiv").offsetWidth;

    var input_margin = {top:10, bottom:10, left:10, right:10};

    var inputSvgHeight =div_height- layerHeight;
    var inputSvgWidth = div_width/5;

    var input_width = +inputSvgWidth- input_margin.left-input_margin.right;
    var input_height = +inputSvgHeight -input_margin.top- input_margin.bottom;

    d3.select('#HiddenLayerDiv').selectAll(".query_image").remove();

    if(selectedDot_index==null){return}

    //console.log(selectedDot_index);


    var svg = d3.select('#HiddenLayerDiv').select("#image_svg");

    var tooltip = d3.select("#input_image").append('div').attr('class', 'hidden image_tooltip');

    svg.selectAll(".query_image").data([0]).enter().append("svg:image")
        .attr("xlink:href","static/data/fashion-mnist/test-images/"+selectedDot_index+".png")
        .attr("class","query_image")
        .attr("width",input_height/4-input_margin.top)
        .attr("height",input_height/4-input_margin.top)
        .attr("x",25)
        .attr("y",input_height/4)
        .on("mouseover",function(d,i){
            tooltip.classed("hidden",false)
            .style("top","400px")
            .text(classes_n[tsne_data[selectedDot_index].label]);
            })
        .on("mouseout",function(d,i){
            tooltip.classed("hidden",true);
            })
        .on("click",function(){
            SubmitInstanceData(selectedDot_index);
        });


}

function DrawHiddenLayer_Query(data_id,data){
    var margin = {top:10, bottom:20, left:25, right:10};
    var svgHeight = document.getElementById("HiddenLayerDiv").offsetHeight/2;
    var svgWidth = (document.getElementById("HiddenLayerDiv").offsetWidth*0.6-svgHeight)/2;
    var width = +svgWidth- margin.left-margin.right;
    var height = +svgHeight -margin.top- margin.bottom;
    var modelID_list =["hiddenlayer_model5","hiddenlayer_model6"];
    //console.log(data);
    //console.log(data_id,data);
    // d3.select('#'+modelID).selectAll(".QueryHiddenLayer").remove();

    if(data==undefined){
        // console.log("Reset!");
        ResetBarChart(modelID_list[0]);
        ResetBarChart(modelID_list[1]);
        return
    }

    if(data_id == "f1" || data_id=="o"){
        DrawBarChart_Query(modelID_list[0],data.v1);
        DrawBarChart_Query(modelID_list[1],data.v2);
    }
    else if(data_id=='c1'){
        DrawConv_Query(data.v1,12,2,4);

    }
    else if(data_id== "c2"){
        DrawConv_Query(data.v2,4,4,10);
    }

    function ResetBarChart(modelID){
        var color = scales_hiddenLayerbar[modelID+data_id].c;
        var svg = d3.select('#'+modelID).select("."+data_id).select(".inner_space");
        svg.selectAll(".query_bar").remove();
        if(data_id!='o'){
            svg.selectAll(".nn_bar").style("fill",function(d){return color(d);}).style('opacity',1);
        }
        else{
            color = d3.scaleOrdinal(d3.schemeCategory10);
            svg.selectAll(".nn_bar").style("fill",function(d,i){return color(i);}).style('opacity',1);
        }



    }

    function DrawBarChart_Query(modelID,plot_data){

        var x = scales_hiddenLayerbar[modelID+data_id].x;
        var y = scales_hiddenLayerbar[modelID+data_id].y;

        var min = d3.extent(plot_data)[0];
        var max = d3.extent(plot_data)[1];

        var max_abs = Math.max(Math.abs(min),Math.abs(max));
        var min_abs = max_abs*(-1.0);

        var color = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);

        var svg = d3.select('#'+modelID).select("."+data_id).select(".inner_space");

        svg.selectAll(".query_bar").remove();
        svg.selectAll(".nn_bar").style("fill",function(){return '#808080'}).style('opacity',0.2);


        var bars =  svg.selectAll(".query_bar")
            .data(plot_data)
            .enter().append("rect")
            .attr("class","query_bar");


      bars.attr("x", function(d) { return x(Math.min(0, d)); })
          .attr("y", function(d,i) { return y(i); })
          .attr("width", function(d) { return Math.abs(x(d) - x(0)); })
          .attr("height", y.bandwidth());

        if(data_id!='o'){
            bars.style("fill",function(d){return color(d);});
        }
        else{
            color = d3.scaleOrdinal(d3.schemeCategory10);
            bars.style("fill",function(d,i){return color(i);});

        }
    }


////This is not tested yet
    function DrawConv_Query(plot_data,rectSize,col_num,gridSize){
        var min = d3.extent(plot_data)[0];
        var max = d3.extent(plot_data)[1];

        var max_abs = Math.max(Math.abs(min),Math.abs(max));
        var min_abs = max_abs*(-1.0);

        var color = d3.scaleLinear().domain([min_abs,min_abs/3.0*2.0,min_abs/3.0,0.0,max_abs/3.0,max_abs/3.0*2.0,max_abs]).range(['#d73027','#fc8d59','#fee090','#ffffbf','#e0f3f8','#91bfdb','#4575b4']);


        var row_size = rectSize*col_num+1;

        var height_size = plot_data.length/rectSize*col_num+ (plot_data.length/(rectSize*col_num))/rectSize-1;


        var svg = d3.select('#'+modelID)
            .append("svg")
            .attr("id",data_id)
            .attr("class","search QueryHiddenLayer")
            .attr("width", svgWidth)
            .attr("height", svgHeight)
            .append("g")
            .attr("transform","translate("+margin.left+","+margin.top+")");

         svg.append("text")
          .attr("transform", "rotate(-90)")
          .attr("y", 0 - margin.left)
          .attr("x",0 - (height / 2))
          .attr("dy", "1em")
          .style("text-anchor", "middle")
          .style("fill","black")
          .text(data_id);


        var conv_rects = svg.selectAll(".query_conv")
                            .data(plot_data)
                            .enter().append("rect")
                            .attr("class","query_conv");


        conv_rects.attr("x",function(d,i){
                    var index_rect = Math.floor(i/(rectSize**2));
                    var col_in_rect = (i%(rectSize**2))%rectSize;
                    var col_index = index_rect%col_num;
                    return (col_index)*(rectSize+1)*gridSize +col_in_rect*gridSize;
                    })
                  .attr("y",function(d,i){
                    var index_rect = Math.floor(i/(rectSize**2));
                    var row_in_rect = Math.floor((i%(rectSize**2))/rectSize);
                    var row_index = Math.floor(index_rect/col_num);
                    return (row_index)*(rectSize+1)*gridSize + row_in_rect*gridSize;
                    })
                  .attr("width",gridSize)
                  .attr("height",gridSize)
                  .style("fill",function(d,i){return color(d);});
        }
}



$(document).ready(function(){
    createStreamGraph();
    DrawLegend();
    createScatterPlot(tsne_data);

});


