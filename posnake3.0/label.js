let video;
let poseNet;
let pose;
let skeleton;
let brain;

let state = 'waiting';
let targetLabel;

//CAP VIDEO TO LABEL
function cap(){
  setTimeout(function() {
      console.log('collecting');
      state = 'collecting';
      setTimeout(function() {
        console.log('not collecting');
        state = 'waiting';
      }, 2000);
    }, 1000);
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  
  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options); 
  
 //LABEL POESES 
  readyButton = createButton('ready');
  readyButton.mousePressed(function(){
   targetLabel = 'ready';
   console.log(targetLabel);
   cap();
  });
  
  upButton = createButton('up');
  upButton.mousePressed(function(){
   targetLabel = 'up';
   console.log(targetLabel);
   cap();
  });

  downButton = createButton('down');
  downButton.mousePressed(function(){
   targetLabel = 'down';
   console.log(targetLabel);
   cap();
  });

  leftButton = createButton('left');
  leftButton.mousePressed(function(){
   targetLabel = 'left';
   console.log(targetLabel);
   cap();
  });

  rightButton = createButton('right');
  rightButton.mousePressed(function(){
   targetLabel = 'right';
   console.log(targetLabel);
   cap();
  });

  saveButton = createButton('save');
  saveButton.mousePressed(function(){
   brain.saveData();
  });
}

//CHECK POSENET LOADED
function modelLoaded() {
  console.log('poseNet Loaded!');
}

function gotPoses(poses){
  if (poses.length > 0){
    pose = poses[0].pose;
    skeleton = poses[0].skeleton; 
    if(state == 'collecting'){
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++){
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1,1);
  image( video, 0, 0, video.width, video.height);
  
  if(pose){
    
    for (let i = 0; i < pose.keypoints.length; i++){
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0,255,0)
      ellipse(x,y,16,16);
    }
    
    for (let i = 0; i < skeleton.length; i++){
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(255);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
  }
}