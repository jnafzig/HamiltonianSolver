mpld3.register_plugin("animatependulum", PendulumAnimatePlugin);
PendulumAnimatePlugin.prototype = Object.create(mpld3.Plugin.prototype);
PendulumAnimatePlugin.prototype.constructor = PendulumAnimatePlugin;
PendulumAnimatePlugin.prototype.requiredProps = ["idpendulum","idtrajectory"];
PendulumAnimatePlugin.prototype.defaultProps = {}
function PendulumAnimatePlugin(fig, props){
    mpld3.Plugin.call(this, fig, props);
};

PendulumAnimatePlugin.prototype.draw = function(){
  var pendulum = mpld3.get_element(this.props.idpendulum);
  var trajectory = mpld3.get_element(this.props.idtrajectory);
  var data = [1,-1,0,0];
  const l1 = 1
  const l2 = 1
  const m1 = 1
  const m2 = 1
  const g = 10
  const dt = 0.02

  function add_mult(x1,x2,a){
    var result = [];
    for(var i=0;i<4;i++) result.push(x1[i]+a*x2[i]);
    return result;
  }

  function pendulum_eom(l1,l2,m1,m2,g,x) {
    let [q1,q2,p1,p2] = x;
    let qdiff = q1-q2;
    let cosdiff = Math.cos(qdiff);
    let sindiff = Math.sin(qdiff);
    let qdenom = m1 + m2*sindiff**2;
    let qdot1 = (l2*p1 - l1*p2*cosdiff)/(l1**2*l2*qdenom);
    let qdot2 = (l1*(m1+m2)*p2 - l2*m2*p1*cosdiff)/(l1*l2**2*qdenom);
    let c1 = p1*p2*sindiff/(l1*l2*qdenom);
    let c2 = (l2**2*m2*p1**2 + l1**2*(m1+m2)*p2**2 - 2*l1*l2*m2*p1*p2*cosdiff)*Math.sin(2*qdiff)/(2*l1**2*l2**2*qdenom**2);
    let pdot1 = -(m1+m2)*g*l1*Math.sin(q1) - c1 + c2;
    let pdot2 = -m2*g*l2*Math.sin(q2) + c1 - c2;
    return [qdot1,qdot2,pdot1,pdot2];
  };

  function rk4_step(time_derivative,h,x) {
    let k1 = time_derivative(x);
    let x1 = add_mult(x, k1, h/2);
    let k2 = time_derivative(x1);
    let x2 = add_mult(x, k2, h/2);
    let k3 = time_derivative(x2);
    let x3 = add_mult(x, k3, h);
    let k4 = time_derivative(x3);

    var result = [];
    for (var i=0;i<4;i++) {
      result.push(h/6 * (k1[i] + k2[i]*2 + k3[i]*2 + k4[i]));
    }
    return result
  };

  var delta_state = rk4_step.bind(null,pendulum_eom.bind(null,l1,l2,m1,m2,g),dt)

  function animate() {

    var data_change
    data_change = delta_state(data);
    data[0] += data_change[0];
    data[1] += data_change[1];
    data[2] += data_change[2];
    data[3] += data_change[3];
    x1 = Math.sin(data[0]);
    y1 = -Math.cos(data[0]);
    x2 = x1 + Math.sin(data[1]);
    y2 = y1 - Math.cos(data[1]);
    pendulum.data = [[0,0],[x1,y1],[x2,y2]];

    // This is doing some sort of unnecessary linear interpolation but I don't
    // enough mpld3.js or d3.js to get it to update without it
    pendulum.elements().transition()
        .attr("d", pendulum.datafunc(pendulum.data))
        .duration(15);

    trajectory.data.push([x2,y2]);

    // This probably needs to be changed so that only the new data is updated:
    trajectory.elements().transition()
        .attr("d", trajectory.datafunc(trajectory.data))
        .duration(15);

  }

  id = setInterval(animate,15)

  var fig = this.fig;
  var coords = fig.canvas.append("text").attr("class", "mpld3-coordinates").style("text-anchor", "end").style("font-size", this.props.fontsize).attr("x", this.fig.width - 5).attr("y", this.fig.height - 5);
  var update_coords = function() {
      var pos = d3.mouse(this), x = fig.axes[1].x.invert(pos[0]), y = fig.axes[1].y.invert(pos[1]);
      trajectory.data = []
      data = [x,y,0,0];
      coords.text("(" + x + ", " + y + ")");
  };
  fig.axes[1].baseaxes.on('mousedown', update_coords);



};
