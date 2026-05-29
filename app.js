const MODEL_BASE_URL = "https://teachablemachine.withgoogle.com/models/LxVJHksKP/";
const PASS_THRESHOLD = 0.6;
const FALLBACK_LABELS = [
  "발모아구르기 준비",
  "발모아구르기 연속동작 1",
  "발모아구르기 연속동작 2",
  "발모아구르기 연속동작 3",
  "발모아구르기 종료",
];

const cameraButton = document.querySelector("#cameraButton");
const measureButton = document.querySelector("#measureButton");
const cameraSelect = document.querySelector("#cameraSelect");
const statusText = document.querySelector("#statusText");
const resultList = document.querySelector("#resultList");
const cameraMount = document.querySelector("#cameraMount");
const poseCanvas = document.querySelector("#poseCanvas");
const poseContext = poseCanvas.getContext("2d");

let model;
let videoElement;
let mediaStream;
let labels = FALLBACK_LABELS;
let isCameraReady = false;
let isMeasuring = false;
let rafId = null;
let analysisErrorCount = 0;

renderRows(labels);

cameraButton.addEventListener("click", async () => {
  if (isCameraReady) return;

  cameraButton.disabled = true;
  cameraButton.textContent = "준비 중";
  statusText.textContent = "모델과 카메라를 준비하고 있습니다.";

  try {
    assertCameraCanStart();
    await init();
    isCameraReady = true;
    cameraButton.textContent = "카메라 실행 중";
    cameraSelect.disabled = true;
    measureButton.disabled = false;
    statusText.textContent = "카메라가 켜졌습니다. 측정 시작을 누르면 판정을 시작합니다.";
  } catch (error) {
    console.error(error);
    cameraButton.disabled = false;
    cameraButton.textContent = "카메라 다시 시작";
    statusText.textContent = getCameraErrorMessage(error);
  }
});

measureButton.addEventListener("click", () => {
  if (!isCameraReady) return;

  if (isMeasuring) {
    stopMeasurement();
  } else {
    startMeasurement();
  }
});

async function init() {
  try {
    await tf.setBackend("webgl");
  } catch (e) {
    console.warn("WebGL backend unavailable, falling back to CPU.", e);
    await tf.setBackend("cpu");
  }
  await tf.ready();

  const modelURL = `${MODEL_BASE_URL}model.json`;
  const metadataURL = `${MODEL_BASE_URL}metadata.json`;
  const metadata = await fetch(metadataURL).then((response) => response.json());

  labels = Array.isArray(metadata.labels) && metadata.labels.length > 0 ? metadata.labels : FALLBACK_LABELS;
  renderRows(labels);
  model = await tmPose.load(modelURL, metadataURL);

  videoElement = await setupCamera();
  cameraMount.innerHTML = "";
  cameraMount.appendChild(videoElement);
  resizePoseCanvas();
}

async function loop() {
  if (!isMeasuring) return;

  try {
    await predict();
  } catch (error) {
    console.error(error);
    analysisErrorCount += 1;

    if (analysisErrorCount >= 20) {
      stopMeasurement();
      statusText.textContent = `동작 분석을 계속할 수 없습니다: ${getAnalysisErrorMessage(error)}`;
      return;
    }
  }

  rafId = window.requestAnimationFrame(loop);
}

async function predict() {
  if (
    !videoElement ||
    videoElement.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA ||
    videoElement.videoWidth === 0 ||
    videoElement.videoHeight === 0
  ) {
    return;
  }

  const { pose, posenetOutput } = await model.estimatePose(videoElement);
  if (!posenetOutput) return;

  const prediction = await model.predict(posenetOutput);
  analysisErrorCount = 0;
  updateRows(prediction);
  drawPose(pose);
}

function startMeasurement() {
  isMeasuring = true;
  measureButton.textContent = "측정 중지";
  measureButton.classList.add("is-measuring");
  statusText.textContent = "측정 중입니다. 동작을 수행하세요.";
  rafId = window.requestAnimationFrame(loop);
}

function stopMeasurement() {
  isMeasuring = false;
  measureButton.textContent = "측정 시작";
  measureButton.classList.remove("is-measuring");
  statusText.textContent = "측정을 중지했습니다. 마지막 결과를 고정해 표시합니다.";

  if (rafId) {
    window.cancelAnimationFrame(rafId);
    rafId = null;
  }
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

async function setupCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  const { width, height } = getCameraSize();
  const facingMode = cameraSelect.value || "environment";
  cameraMount.classList.toggle("is-front", facingMode === "user");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: width },
      height: { ideal: height },
    },
  });

  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = mediaStream;

  await video.play();
  await waitForVideoReady(video);
  return video;
}

function waitForVideoReady(video) {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => reject(new Error("VIDEO_READY_TIMEOUT")), 7000);

    const checkReady = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        window.clearTimeout(timeoutId);
        resolve();
      }
    };

    video.addEventListener("loadedmetadata", checkReady, { once: true });
    video.addEventListener("loadeddata", checkReady, { once: true });
    video.addEventListener("canplay", checkReady, { once: true });
    checkReady();
  });
}

function assertCameraCanStart() {
  if (!window.isSecureContext) {
    throw new Error("INSECURE_CONTEXT");
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("CAMERA_UNSUPPORTED");
  }
}

function getCameraErrorMessage(error) {
  const name = error?.name || error?.message;

  if (name === "INSECURE_CONTEXT") {
    return "HTTPS 주소에서만 카메라를 사용할 수 있습니다. GitHub Pages 주소로 접속하세요.";
  }

  if (name === "CAMERA_UNSUPPORTED") {
    return "이 브라우저에서는 카메라 API를 지원하지 않습니다. Chrome 또는 Safari로 접속하세요.";
  }

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "카메라 권한이 차단되었습니다. 브라우저 주소창의 사이트 설정에서 카메라를 허용하세요.";
  }

  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "사용 가능한 카메라를 찾지 못했습니다. 기기의 카메라 사용 가능 상태를 확인하세요.";
  }

  if (name === "NotReadableError" || name === "TrackStartError") {
    return "다른 앱이 카메라를 사용 중일 수 있습니다. 카메라 앱이나 화상회의 앱을 종료하세요.";
  }

  return "카메라를 시작하지 못했습니다. 브라우저 권한과 HTTPS 접속을 확인하세요.";
}

function getAnalysisErrorMessage(error) {
  const name = error?.name || "";
  const message = error?.message || "알 수 없는 오류";

  if (name === "VIDEO_READY_TIMEOUT" || message === "VIDEO_READY_TIMEOUT") {
    return "카메라 영상 정보를 읽지 못했습니다. 새로고침 후 다시 시도하세요.";
  }

  if (message.includes("fromPixels")) {
    return "이미지 처리 함수(tf.browser.fromPixels)를 사용할 수 없습니다. 브라우저를 최신 버전으로 업데이트하거나 Chrome으로 접속해 보세요.";
  }

  if (message.includes("WebGL") || name.includes("WebGL")) {
    return "브라우저의 WebGL 가속 문제일 수 있습니다. Chrome 또는 Safari를 다시 실행하세요.";
  }

  return `${name ? name + ": " : ""}${message}`;
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }

  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

window.addEventListener("resize", resizePoseCanvas);
