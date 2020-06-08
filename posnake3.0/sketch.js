let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "loading...";

let state = 'waiting';
let targetLabel;

let snake;
let rez = 20;
let food;
let w;
let h;


function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  
  //SETTING
  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  
  // LOAD PRETRAINED MODEL
  const modelInfo = {
    model: 'model.json',
    metadata: 'model_meta.json',
    weights: 'model.weights.bin',
  };
  brain.load(modelInfo, brainLoaded);
  
  //SNAKE GAME
  w = floor(width / rez);
  h = floor(height / rez);
  frameRate(5);
  snake = new Snake();
  foodLocation();
}

//CHECK NEURALNETWORK LOADED
function brainLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

//CLASSIFY
function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

//SNAKE FOOD LOCATION
function foodLocation() {
  let x = floor(random(w));
  let y = floor(random(h));
  food = createVector(x, y);
}

// CONTROL SNAKE BY POSELABEL
function controlSnake() {
  if (poseLabel === "left") {
    snake.setDir(-1, 0);
  } else if (poseLabel === "right") {
    snake.setDir(1, 0);
  } else if (poseLabel === "down") {
    snake.setDir(0, 1);
  } else if (poseLabel === "up") {
    snake.setDir(0, -1);
  }else if (poseLabel === "ready") {
    snake.setDir(0, -1);
  }
}

function gotResult(error, results) {  
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label;
    controlSnake();
  }else if (error) {
    console.error(error);
    return;
  }
  classifyPose();
  controlSnake();
}

//COLLECT POSEDATA FROM WEBCAM
function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
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

//CHECK POSENET
function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1,1);
  image( video, 0, 0, video.width, video.height);
  
//WEBCAM WITH SKELETON
  if(pose){ 
    for (let i = 0; i < pose.keypoints.length; i++){
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(255, 206, 136)
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
  pop();
  noStroke();
  textSize(40);
  fill(255);
  text(poseLabel, 10, 50);
  

//SNAKE GAME
  scale(rez);
  
  if (snake.eat(food)) {
    foodLocation();
  }
  snake.update();
  snake.show();


  if (snake.endGame()) {
    print("END GAME");
    background(255, 0, 0);
    noLoop();
  }

  noStroke();
  fill(255, 0, 0);
  rect(food.x, food.y, 1, 1);
}


