const origin = [$(window).width() / 2, $(window).height() / 1.4];
const j = 10;
const scale = 20;
var keypoints = [], line_data = [], xGrid_data = [], yGrid1_data = [], yGrid2_data = [];

var svg = d3.select('svg.main').append('g').attr("class", "stickman");

var frameNumber;
var mx, my, mouseX, mouseY, beta = 0, alpha = 0, startAngle = Math.PI / 4;

const parent_joints = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15];
const joints_left = [4, 5, 6, 11, 12, 13];
const joints_right = [1, 2, 3, 14, 15, 16];

//mimic python range() method
let range = n => Array.from(Array(n).keys());

let grid3dx = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

let grid3dy1 = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

let grid3dy2 = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

let point3d = d3._3d()
    .x(function (d) {
        return d.x;
    })
    .y(function (d) {
        return d.y;
    })
    .z(function (d) {
        return d.z;
    })
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale * 10);

let line3d = d3._3d()
    .shape('LINE')
    .x(function (d) {
        return d.x;
    })
    .y(function (d) {
        return d.y;
    })
    .z(function (d) {
        return d.z;
    })
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale * 10);


function createXGrid(data) {
    var xGrid = svg.selectAll('path.xgrid').data(data);

    xGrid
        .enter()
        .append('path')
        .attr('class', '_3d xgrid')
        .merge(xGrid)
        .attr('stroke', 'black')
        .attr('stroke-width', 0.3)
        .attr('fill', function (d) {
            return d.ccw ? 'lightgrey' : '#717171';
        })
        .attr('fill-opacity', 0.9)
        .attr('d', grid3dx.draw);

    xGrid.exit().remove();
}

function createYGrid1(data) {
    var yGrid1 = svg.selectAll('path.ygrid1').data(data);

    yGrid1
        .enter()
        .append('path')
        .attr('class', '_3d ygrid1')
        .merge(yGrid1)
        .attr('stroke', 'black')
        .attr('stroke-width', 0.3)
        .attr('fill', function (d) {
            return d.ccw ? 'lightgrey' : '#717171';
        })
        .attr('fill-opacity', 0.9)
        .attr('d', grid3dy1.draw);

    yGrid1.exit().remove();
}

function createYGrid2(data) {
    var yGrid2 = svg.selectAll('path.ygrid2').data(data);

    yGrid2
        .enter()
        .append('path')
        .attr('class', '_3d ygrid2')
        .merge(yGrid2)
        .attr('stroke', 'black')
        .attr('stroke-width', 0.3)
        .attr('fill', function (d) {
            return d.ccw ? 'lightgrey' : '#717171';
        })
        .attr('fill-opacity', 0.9)
        .attr('d', grid3dy2.draw);

    yGrid2.exit().remove();
}

function createPoints(data) {
    var points = svg.selectAll('circle').data(data);

    points
        .enter()
        .append('circle')
        .attr('class', '_3d')
        .attr('opacity', 0)
        .attr('cx', function (d) {
            return d.projected.x
        })
        .attr('cy', function (d) {
            return d.projected.y
        })
        .merge(points)
        .attr('r', 3)
        .attr('stroke', function (d) {
            return d3.color('red');
        })
        .attr('fill', function (d) {
            return d3.color('red');
        })
        .attr('opacity', 1)
        .attr('cx', function (d) {
            return d.projected.x
        })
        .attr('cy', function (d) {
            return d.projected.y
        });

    points.exit().remove();
}

function createLines(data) {
    var lines = svg.selectAll('line').data(data);

    lines
        .enter()
        .append('line')
        .attr('class', '_3d')
        .attr('stroke-width', 1)
        .merge(lines)
        .attr('fill', function (d) {
            return d3.color(d[0].colour);
        })
        .attr('stroke', function (d) {
            return d3.color(d[0].colour);
        })
        .attr('x1', function (d) {
            return d[0].projected.x;
        })
        .attr('y1', function (d) {
            return d[0].projected.y;
        })
        .attr('x2', function (d) {
            return d[1].projected.x;
        })
        .attr('y2', function (d) {
            return d[1].projected.y;
        });

    lines.exit().remove();
}

function processData(data) {

    createXGrid(data[0]);

    createYGrid1(data[2]);

    createYGrid2(data[3]);

    createPoints(data[1]);

    createLines(data[4]);

    d3.selectAll('._3d').sort(d3._3d().sort);
}


function drawStickman() {
    // create grid data
    xGrid_data = [], yGrid1_data = [], yGrid2_data = [];
    for (var z = -j; z < j; z++) {
        for (var x = -j; x < j; x++) {
            xGrid_data.push([x, 0, z]);
            yGrid1_data.push([9, x - 9, z]);
            yGrid2_data.push([x, z - 9, -10]);

        }
    }


    // create keypoints data
    keypoints = [];
    swing_keypoints[frameNumber].forEach((element) => {
        keypoints.push({
            x: element[0],
            y: element[2],
            z: element[1],
        })
    });

    // create line data
    line_data = [];
    for (const [index, start_point] of swing_keypoints[frameNumber].entries()) {
        let parent = parent_joints[index];
        if (parent !== -1) {
            let end_point = swing_keypoints[frameNumber][parent];
            let line_colour = (joints_left.includes(index) || joints_left.includes(parent)) ? 'blue' : 'black';
            line_data.push(
                [
                    {
                        x: start_point[0],
                        y: start_point[2],
                        z: start_point[1],
                        colour: line_colour,
                    },
                    {
                        x: end_point[0],
                        y: end_point[2],
                        z: end_point[1],
                    }
                ]
            )
        }

    }

    let data = [
        grid3dx(xGrid_data),
        point3d(keypoints),
        grid3dy1(yGrid1_data),
        grid3dy2(yGrid2_data),
        line3d(line_data),
    ];
    processData(data);
}

function dragStart() {
    mx = d3.event.x;
    my = d3.event.y;
}

function dragged() {
    mouseX = mouseX || 0;
    mouseY = mouseY || 0;
    beta = (d3.event.x - mx + mouseX) * Math.PI / 230;
    alpha = (d3.event.y - my + mouseY) * Math.PI / 230 * (-1);
    let data = [
        grid3dx.rotateY(beta + startAngle).rotateX(alpha - startAngle)(xGrid_data),
        point3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(keypoints),
        grid3dy1.rotateY(beta + startAngle).rotateX(alpha - startAngle)(yGrid1_data),
        grid3dy2.rotateY(beta + startAngle).rotateX(alpha - startAngle)(yGrid2_data),
        line3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(line_data),
    ];
    processData(data);
}

function dragEnd() {
    mouseX = d3.event.x - mx + mouseX;
    mouseY = d3.event.y - my + mouseY;
}

var swing_keypoints = [];
$.getJSON('data.json', function (data) {
    swing_keypoints = data;

    frameNumber = 0;
    drawStickman();

    sliderInit(data.length);
    // drag stick man
    d3.select("g.stickman").call(d3.drag().on('drag', dragged).on('start', dragStart).on('end', dragEnd));

});


function sliderInit(sliderSteps) {
    var sliderStep = d3
    .sliderBottom()
    .min(d3.min(range(sliderStep)))
    .max(d3.max(range(sliderSteps)))
    .width(300)
    .step(1)
    .default(0)
    .on('onchange', val => {
      frameNumber = val;
      drawStickman();
    });

    const slider = d3
        .select('div#slider-step')
        .append('svg')
        .attr('class', 'slider')
        .attr('width', 500)
        .attr('height', 100)
        .append('g')
        .attr('class', 'slider')
        .attr('transform', 'translate(30,30)');

    // drag slider
    slider.call(sliderStep);
}