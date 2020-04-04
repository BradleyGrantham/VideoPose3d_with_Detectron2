var origin = [$(window).width() / 2, $(window).height() / 1.4], j = 10, scale = 20,
    keypoints = [], yLine = [], xGrid = [],
    yGrid1 = [], yGrid2 = [], line_data = [], beta = 0, alpha = 0, key = function (d) {
        return d.id;
    }, startAngle = 0;
var svg = d3.select('svg').call(d3.drag().on('drag', dragged).on('start', dragStart).on('end', dragEnd)).append('g');
var color = d3.scaleOrdinal(d3.schemeCategory20);
var mx, my, mouseX, mouseY;

const golf_swing = [
    [
      -3.211162402283661e-05,
      7.152739999582991e-05,
      -0.8912246823310852
    ],
    [
      0.02407112531363976,
      0.09900563210248947,
      -0.8931843042373657
    ],
    [
      0.04991847276687628,
      0.10360199213027954,
      -0.4480474293231964
    ],
    [
      0.0078020691871643144,
      0.09741427004337311,
      -0.014388442039489744
    ],
    [
      -0.024060048162937064,
      -0.09900061041116714,
      -0.8891841173171997
    ],
    [
      -0.02230921387672419,
      -0.061637260019779205,
      -0.4462723135948181
    ],
    [
      -0.09529447555541992,
      -0.02673597447574138,
      -0.014570891857147229
    ],
    [
      -0.05386868119239793,
      0.03230265900492668,
      -1.122077465057373
    ],
    [
      -0.06931942701339705,
      0.09160071611404419,
      -1.3659696578979492
    ],
    [
      0.008772790431976502,
      0.1012304350733757,
      -1.4406342506408691
    ],
    [
      -0.06401711702346782,
      0.13830940425395966,
      -1.5252538919448853
    ],
    [
      -0.10293233394622786,
      -0.036846190690994256,
      -1.338482141494751
    ],
    [
      0.025293141603469987,
      -0.2650742530822754,
      -1.2616517543792725
    ],
    [
      0.1646957993507387,
      -0.1577398180961609,
      -1.4385120868682861
    ],
    [
      -0.008822530508041208,
      0.18670329451560974,
      -1.3169260025024414
    ],
    [
      0.2024367302656175,
      0.16720521450042725,
      -1.1961545944213867
    ],
    [
      0.3134011030197145,
      0.01506897807121275,
      -1.2654986381530762
    ]
  ];

const parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15];

const joints_left = [4, 5, 6, 11, 12, 13];
const joints_right = [1, 2, 3, 14, 15, 16];

var grid3dx = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

var grid3dy1 = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

var grid3dy2 = d3._3d()
    .shape('GRID', 20)
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);

var point3d = d3._3d()
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


var line3d = d3._3d()
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

function processData(data, tt) {

    /* ----------- GRID ----------- */

    var xGrid = svg.selectAll('path.xgrid').data(data[0], key);

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

    var yGrid1 = svg.selectAll('path.ygrid1').data(data[2], key);

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


    var yGrid2 = svg.selectAll('path.ygrid2').data(data[3], key);

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

    /* ----------- POINTS ----------- */

    var points = svg.selectAll('circle').data(data[1], key);

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
        .transition().duration(tt)
        .attr('r', 3)
        .attr('stroke', function (d) {
            return d3.color(color(d.id)).darker(3);
        })
        .attr('fill', function (d) {
            return color(d.id);
        })
        .attr('opacity', 1)
        .attr('cx', posPointX)
        .attr('cy', posPointY);

    points.exit().remove();

    /* ----------- lines ----------- */

    var lines = svg.selectAll('line').data(data[4]);

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

    d3.selectAll('._3d').sort(d3._3d().sort);
}


function init() {
    var cnt = 0;
    xGrid = [], yGrid1 = [], yGrid2 = [], keypoints = [], yLine = [];
    for (var z = -j; z < j; z++) {
        for (var x = -j; x < j; x++) {
            xGrid.push([x, 0, z]);
            yGrid1.push([9, x - 9, z]);
            yGrid2.push([x, z - 9, -10]);

        }
    }

    golf_swing.forEach((element) => {
        keypoints.push({
            x: element[0],
            y: element[2],
            z: element[1],
            id: 'point_' + cnt++
        })
    });

    line_data = [];
    for (const [index, start_point] of golf_swing.entries()) {
        var parent = parents[index];
        if (parent !== -1) {
            var end_point = golf_swing[parent];
            var line_colour = (joints_left.includes(index) || joints_left.includes(parent)) ? 'red' : 'black';
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

    var data = [
        grid3dx(xGrid),
        point3d(keypoints),
        grid3dy1(yGrid1),
        grid3dy2(yGrid2),
        line3d(line_data),
    ];
    processData(data, 1000);
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
    var data = [
        grid3dx.rotateY(beta + startAngle).rotateX(alpha - startAngle)(xGrid),
        point3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(keypoints),
        grid3dy1.rotateY(beta + startAngle).rotateX(alpha - startAngle)(yGrid1),
        grid3dy2.rotateY(beta + startAngle).rotateX(alpha - startAngle)(yGrid2),
        line3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(line_data),
    ];
    processData(data, 0);
}

function dragEnd() {
    mouseX = d3.event.x - mx + mouseX;
    mouseY = d3.event.y - my + mouseY;
}

d3.selectAll('button').on('click', init);

init();
