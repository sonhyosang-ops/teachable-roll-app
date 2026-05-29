const MODEL_BASE_URL = "https://teachablemachine.withgoogle.com/models/LxVJHksKP/";
const PASS_THRESHOLD = 0.6;
const FALLBACK_LABELS = [
  "발모아구르기 준비",
  "발모아구르기 연속동작 1",
  "발모아구르기 연속동작 2",
  "발모아구르기 연속동작 3",
  "발모아구르기 종료",
];

const startButton = document.querySelector("#startButton");
const statusText = document.querySelector("#statusText");
const resultList = document.querySelector("#resultList");
const cameraMount = document.querySelector("#cameraMount");
const poseCanvas = document.querySelector("#poseCanvas");
const poseContext = poseCanvas.getContext("2d");

let model;
let webcam;
let labels = FALLBACK_LABELS;
let isRunning = false;

renderRows(labels);

startButton.addEventListener("click", async () => {
  if (isRunning) return;

  startButton.disabled = true;
  startButton.textContent = "준비 중";
  statusText.textContent = "모델과 카메라를 준비하고 있습니다.";

  try {
    await init();
    isRunning = true;
    startButton.textContent = "인식 중";
    statusText.textContent = "카메라 앞에서 동작을 수행하세요.";
    window.requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = "다시 시작";
    statusText.textContent = "카메라 권한 또는 모델 불러오기를 확인하세요.";
  }
});

async function init() {
  const modelURL = `${MODEL_BASE_URL}model.json`;
  const metadataURL = `${MODEL_BASE_URL}metadata.json`;
  const metadata = await fetch(metadataURL).then((response) => response.json());

  labels = Array.isArray(metadata.labels) && metadata.labels.length > 0 ? metadata.labels : FALLBACK_LABELS;
  renderRows(labels);
  model = await tmPose.load(modelURL, metadataURL);

  const { width, height } = getCameraSize();
  webcam = new tmPose.Webcam(width, height, false);
  await webcam.setup({ facingMode: "user" });
  await webcam.play();

  cameraMount.innerHTML = "";
  cameraMount.appendChild(webcam.canvas);
  resizePoseCanvas();
}

async function loop() {
  if (!isRunning) return;

  webcam.update();
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
  const prediction = await model.predict(posenetOutput);
  updateRows(prediction);
  drawPose(pose);
}

function updateRows(predictions) {
  predictions.forEach((prediction) => {
    let row = resultList.querySelector(`[data-motion="${cssEscape(prediction.className)}"]`);

    if (!row) {
      row = createResultRow(prediction.className);
      resultList.appendChild(row);
    }

    const score = prediction.probability;
    const passed = score >= PASS_THRESHOLD;

    row.querySelector(".score").textContent = `${Math.round(score * 100)}%`;
    const badge = row.querySelector(".badge");
    badge.textContent = passed ? "통과" : "노력하세요";
    badge.className = `badge ${passed ? "pass" : "try"}`;
  });
}

function renderRows(rowLabels) {
  resultList.innerHTML = "";

  rowLabels.forEach((label) => {
    resultList.appendChild(createResultRow(label));
  });
}

function createResultRow(label) {
  const row = document.createElement("article");
  row.className = "result-row";
  row.dataset.motion = label;

  const name = document.createElement("div");
  name.className = "motion-name";
  name.textContent = label;

  const score = document.createElement("div");
  score.className = "score";
  score.textContent = "0%";

  const badge = document.createElement("div");
  badge.className = "badge try";
  badge.textContent = "노력하세요";

  row.append(name, score, badge);
  return row;
}

function drawPose(pose) {
  resizePoseCanvas();
  poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

  if (!pose) return;

  const minPartConfidence = 0.5;
  tmPose.drawKeypoints(pose.keypoints, minPartConfidence, poseContext);
  tmPose.drawSkeleton(pose.keypoints, minPartConfidence, poseContext);
}

function resizePoseCanvas() {
  const rect = cameraMount.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  if (poseCanvas.width !== width || poseCanvas.height !== height) {
    poseCanvas.width = width;
    poseCanvas.height = height;
  }
}

function getCameraSize() {
  const rect = cameraMount.getBoundingClientRect();
  return {
    width: Math.max(320, Math.round(rect.width)),
    height: Math.max(240, Math.round(rect.height)),
  };
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }

  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

window.addEventListener("resize", resizePoseCanvas);
