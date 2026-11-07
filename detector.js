const cameraCard = document.getElementById("cameraCard");
    const video = document.getElementById("cameraVideo");
    const scanButton = document.getElementById("scanButton");
    const buttonText = document.getElementById("buttonText");
    const statusTitle = document.getElementById("statusTitle");
    const statusDetail = document.getElementById("statusDetail");
    const faceCount = document.getElementById("faceCount");
    const scanStatus = document.getElementById("scanStatus");
    const notice = document.getElementById("notice");
    const menuButton = document.getElementById("menuButton");
    const mainMenu = document.getElementById("mainMenu");
    const activityDialog = document.getElementById("activityDialog");
    const registrationDialog = document.getElementById("registrationDialog");
    const registrationForm = document.getElementById("registrationForm");
    const registrationScanner = document.getElementById("registrationScanner");
    const registrationScanStatus = document.getElementById("registrationScanStatus");
    const registrationScanButton = document.getElementById("registrationScanButton");
    const registrationScanIndicator = document.getElementById("registrationScanIndicator");
    const registrationCaptureOverlay = document.getElementById("registrationCaptureOverlay");
    let stream = null;
    let detector = null;
    let detectorPromise = null;
    let detectionTimer = null;
    let noticeTimer = null;
    let detectionInProgress = false;
    let scanning = false;
    let scanGeneration = 0;
    let detectedFaceCount = 0;
    let registrationScanStarted = false;
    let registrationCaptureComplete = false;
    let registrationScanTransition = false;
    let registrationPoseIndex = 0;
    let registrationPoseFrames = 0;
    let registrationCalibrationSamples = [];
    let registrationBaselinePose = null;
    const registrationCalibrationFrameCount = 8;
    const registrationPoses = [
      { direction: "LEFT", instruction: "Turn your head LEFT and show your side profile", matches: (pose, baseline) => pose.yaw > baseline.yaw + 0.16 },
      { direction: "RIGHT", instruction: "Turn your head RIGHT and show your side profile", matches: (pose, baseline) => pose.yaw < baseline.yaw - 0.16 },
      { direction: "DOWN", instruction: "Tilt your face DOWN toward your chest", matches: (pose, baseline) => pose.pitch > baseline.pitch + 0.12 },
      { direction: "UP", instruction: "Tilt your face UP toward the ceiling", matches: (pose, baseline) => pose.pitch < baseline.pitch - 0.12 }
    ];

    const faceModelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
    const visionWasmUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm";
    const visionModuleUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";

    const clockTime = document.getElementById("clockTime");
    const clockDate = document.getElementById("clockDate");

    function updateClock() {
      const now = new Date();
      clockTime.textContent = new Intl.DateTimeFormat("en-PH", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      }).format(now);
      clockDate.textContent = new Intl.DateTimeFormat("en-PH", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric"
      }).format(now);
    }

    updateClock();
    window.setInterval(updateClock, 1000);

    function showNotice(message) {
      notice.textContent = message;
      notice.classList.add("show");
      window.clearTimeout(noticeTimer);
      noticeTimer = window.setTimeout(() => notice.classList.remove("show"), 3600);
    }

    function setStatus(state, title, detail) {
      scanStatus.dataset.state = state;
      statusTitle.textContent = title;
      statusDetail.textContent = detail;
      registrationScanStatus.textContent = title;
    }

    function updateRegistrationPose(detections) {
      if (!registrationScanStarted) return;
      if (detections.length !== 1) {
        registrationPoseFrames = 0;
        if (registrationCalibrationSamples.length < registrationCalibrationFrameCount) {
          registrationCalibrationSamples = [];
        }
        registrationScanIndicator.textContent = detections.length > 1
          ? "Only one face at a time. Move extra faces out of frame."
          : "Position one face in the camera frame.";
        return;
      }

      const landmarks = detections[0];
      const leftEye = landmarks[33];
      const rightEye = landmarks[263];
      const nose = landmarks[1];
      const chin = landmarks[152];
      if (!leftEye || !rightEye || !nose || !chin) {
        registrationPoseFrames = 0;
        registrationScanIndicator.textContent = "Face angle unavailable. Face the camera and try again.";
        return;
      }

      const eyeWidth = Math.abs(rightEye.x - leftEye.x);
      const eyeCenterX = (leftEye.x + rightEye.x) / 2;
      const eyeCenterY = (leftEye.y + rightEye.y) / 2;
      const faceHeight = chin.y - eyeCenterY;
      if (![eyeWidth, eyeCenterX, eyeCenterY, nose.x, nose.y, faceHeight].every(Number.isFinite)
        || eyeWidth < 0.01 || faceHeight < 0.01) {
        registrationPoseFrames = 0;
        registrationScanIndicator.textContent = "Face angle unavailable. Face the camera and try again.";
        return;
      }

      const currentPose = {
        yaw: (nose.x - eyeCenterX) / eyeWidth,
        pitch: (nose.y - eyeCenterY) / faceHeight
      };

      if (!registrationBaselinePose) {
        registrationCalibrationSamples.push(currentPose);
        const remaining = registrationCalibrationFrameCount - registrationCalibrationSamples.length;
        if (remaining > 0) {
          registrationScanIndicator.textContent = `Look straight at the camera (${registrationCalibrationSamples.length}/${registrationCalibrationFrameCount})`;
          return;
        }
        registrationBaselinePose = registrationCalibrationSamples.reduce((average, sample) => ({
          yaw: average.yaw + sample.yaw / registrationCalibrationFrameCount,
          pitch: average.pitch + sample.pitch / registrationCalibrationFrameCount
        }), { yaw: 0, pitch: 0 });
        registrationScanIndicator.textContent = `${registrationPoses[0].instruction} (1 of ${registrationPoses.length})`;
        return;
      }

      const pose = registrationPoses[registrationPoseIndex];
      registrationScanIndicator.textContent = `${pose.instruction} (${registrationPoseIndex + 1} of ${registrationPoses.length})`;
      registrationPoseFrames = pose.matches(currentPose, registrationBaselinePose) ? registrationPoseFrames + 1 : 0;

      if (registrationPoseFrames >= 2) {
        registrationPoseIndex += 1;
        registrationPoseFrames = 0;
        if (registrationPoseIndex === registrationPoses.length) {
          finishRegistrationScan();
          return;
        }
        const nextPose = registrationPoses[registrationPoseIndex];
        registrationScanIndicator.textContent = `${nextPose.instruction} (${registrationPoseIndex + 1} of ${registrationPoses.length})`;
      }
    }

    function updateFaces(detections) {
      const count = detections.length;
      detectedFaceCount = count;
      faceCount.textContent = `${count} ${count === 1 ? "face" : "faces"}`;
      if (count > 0) {
        setStatus("scanning", count === 1 ? "Face detected" : `${count} faces detected`, "Scanning continues.");
      } else {
        setStatus("scanning", "Looking for faces", "Center your face in the frame.");
      }
      if (registrationScanStarted) updateRegistrationPose(detections);
    }

    async function initializeDetector() {
      if (detector) return detector;
      if (detectorPromise) return detectorPromise;

      detectorPromise = (async () => {
        let visionError;
        try {
          const { FilesetResolver, FaceLandmarker } = await import(visionModuleUrl);
          const vision = await FilesetResolver.forVisionTasks(visionWasmUrl);
          try {
            detector = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: faceModelUrl, delegate: "GPU" },
              runningMode: "VIDEO",
              numFaces: 2
            });
          } catch {
            detector = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: faceModelUrl, delegate: "CPU" },
              runningMode: "VIDEO",
              numFaces: 2
            });
          }
          return detector;
        } catch (error) {
          visionError = error;
        }

        throw new Error(`Face landmarks could not load for head-pose scanning. Check your internet connection and browser support. ${visionError.message || ""}`.trim());
      })();

      try {
        return await detectorPromise;
      } finally {
        detectorPromise = null;
      }
    }

    async function detectFaces() {
      if (!detector || !scanning || detectionInProgress || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      detectionInProgress = true;
      try {
        const activeDetector = detector;
        const result = activeDetector.detectForVideo(video, performance.now());
        const detections = result.faceLandmarks || [];
        if (scanning && detector === activeDetector) updateFaces(detections);
      } catch (error) {
        stopScan();
        setStatus("error", "Scan stopped", "The camera frame could not be read. Try again.");
        showNotice(`Could not read the camera frame: ${error.message || "unknown error."}`);
      } finally {
        detectionInProgress = false;
      }
    }

    function stopScan(state = "stopped") {
      scanGeneration += 1;
      scanning = false;
      registrationScanStarted = false;
      registrationCaptureComplete = false;
      registrationPoseIndex = 0;
      registrationPoseFrames = 0;
      registrationCalibrationSamples = [];
      registrationBaselinePose = null;
      registrationScanIndicator.hidden = true;
      registrationCaptureOverlay.hidden = true;
      window.clearInterval(detectionTimer);
      detectionTimer = null;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
      cameraCard.classList.remove("camera-on");
      cameraCard.classList.remove("detecting");
      buttonText.textContent = "Start scan";
      scanButton.setAttribute("aria-label", "Start face scan");
      scanButton.disabled = false;
      updateRegistrationScanButton();
      setStatus(state, "Camera paused", "Select the button to resume.");
      faceCount.textContent = "—";
      detectedFaceCount = 0;
    }

    function finishRegistrationScan() {
      if (!scanning || !registrationScanStarted || registrationCaptureComplete) return;
      registrationScanStarted = false;
      registrationCaptureComplete = true;
      scanning = false;
      registrationPoseFrames = 0;
      registrationCalibrationSamples = [];
      registrationBaselinePose = null;
      window.clearInterval(detectionTimer);
      detectionTimer = null;
      cameraCard.classList.remove("detecting");
      registrationScanner.dataset.scanState = "done";
      registrationScanStatus.textContent = "Four-way face scan complete";
      registrationScanIndicator.hidden = true;
      registrationCaptureOverlay.hidden = false;
      setStatus("scanning", "Face captured", "Registration face scan complete.");
      registrationScanStatus.textContent = "Face captured";
      buttonText.textContent = "Stop camera";
      scanButton.setAttribute("aria-label", "Stop camera");
      scanButton.disabled = false;
      updateRegistrationScanButton();
    }

    function updateRegistrationScanButton() {
      registrationScanButton.disabled = registrationCaptureComplete || registrationScanStarted;
      registrationScanButton.textContent = registrationCaptureComplete
        ? "Done"
        : registrationScanStarted
          ? "Scanning…"
          : "Start scanning";
    }

    async function startScan() {
      const activeScan = scanGeneration;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        buttonText.textContent = "Try again";
        scanButton.setAttribute("aria-label", "Try opening the camera again");
        scanButton.disabled = false;
        updateRegistrationScanButton();
        setStatus("error", "Camera unavailable", "Open this page over HTTPS or localhost.");
        showNotice("Camera access requires HTTPS or localhost.");
        return;
      }
      scanButton.disabled = true;
      buttonText.textContent = "Connecting…";
      setStatus("starting", "Requesting permission", "Allow camera access in your browser.");
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "user", width: { ideal: 720 }, height: { ideal: 960 } }
        });
        if (activeScan !== scanGeneration) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
          return;
        }
        video.srcObject = stream;
        await video.play();
        if (activeScan !== scanGeneration) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
          video.srcObject = null;
          return;
        }
        cameraCard.classList.add("camera-on");
        scanning = true;
        scanButton.disabled = false;
        updateRegistrationScanButton();
        buttonText.textContent = "Stop camera";
        scanButton.setAttribute("aria-label", "Stop camera");

        setStatus("starting", "Loading face detector", "Preparing on-device detection. First load needs an internet connection.");
        try {
          await initializeDetector();
          if (!scanning || activeScan !== scanGeneration) return;
          cameraCard.classList.add("detecting");
          setStatus("scanning", "Looking for faces", "Center your face in the frame.");
          buttonText.textContent = "Stop scan";
          scanButton.setAttribute("aria-label", "Stop face scan");
          detectionTimer = window.setInterval(detectFaces, 300);
          detectFaces();
        } catch (error) {
          if (!scanning || activeScan !== scanGeneration) return;
          registrationScanStarted = false;
          registrationScanner.dataset.scanState = "error";
          registrationScanIndicator.hidden = true;
          updateRegistrationScanButton();
          setStatus("unsupported", "Camera preview only", error.message);
          faceCount.textContent = "—";
          detectedFaceCount = 0;
          showNotice("Face detection is unavailable. Check your internet connection or use a supported browser.");
        }
      } catch (error) {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
        }
        video.srcObject = null;
        cameraCard.classList.remove("camera-on", "detecting");
        scanning = false;
        registrationScanStarted = false;
        registrationScanner.dataset.scanState = "error";
        scanButton.disabled = false;
        updateRegistrationScanButton();
        buttonText.textContent = "Try again";
        scanButton.setAttribute("aria-label", "Try opening the camera again");
        const permissionDenied = error.name === "NotAllowedError" || error.name === "PermissionDeniedError";
        const noCamera = error.name === "NotFoundError" || error.name === "DevicesNotFoundError";
        const detail = permissionDenied
          ? "Allow camera access in your browser settings, then try again."
          : noCamera
            ? "No camera was found on this device."
            : "Check camera access and connection, then try again.";
        setStatus("error", "Camera unavailable", detail);
        showNotice(`Could not start the camera: ${error.message || "access was denied or unavailable."}`);
      }
    }

    scanButton.addEventListener("click", () => {
      if (stream) stopScan();
      else startScan();
    });

    registrationScanButton.addEventListener("click", async () => {
      if (registrationCaptureComplete || registrationScanStarted) return;
      registrationScanStarted = true;
      registrationPoseIndex = 0;
      registrationPoseFrames = 0;
      registrationCalibrationSamples = [];
      registrationBaselinePose = null;
      registrationScanner.dataset.scanState = "scanning";
      registrationScanStatus.textContent = "Follow the four camera instructions";
      registrationScanIndicator.hidden = false;
      registrationScanIndicator.textContent = "Position one face in the camera frame.";
      updateRegistrationScanButton();
      registrationScanTransition = true;
      registrationDialog.close("start-scan");

      if (!stream) {
        await startScan();
        if (!scanning) {
          registrationScanStarted = false;
          registrationScanIndicator.hidden = true;
          registrationScanner.dataset.scanState = "error";
          registrationScanStatus.textContent = statusTitle.textContent;
          updateRegistrationScanButton();
        }
        return;
      }

      try {
        await initializeDetector();
        if (!registrationScanStarted) return;
        if (!scanning) {
          scanning = true;
          cameraCard.classList.add("detecting");
          detectionTimer = window.setInterval(detectFaces, 300);
        }
        detectFaces();
      } catch (error) {
        registrationScanStarted = false;
        registrationScanner.dataset.scanState = "error";
        registrationScanIndicator.hidden = true;
        registrationScanStatus.textContent = "Face scanner unavailable";
        updateRegistrationScanButton();
        showNotice(`Face detection is unavailable: ${error.message || "unknown error."}`);
      }
    });

    menuButton.addEventListener("click", () => {
      const isExpanded = menuButton.getAttribute("aria-expanded") === "true";
      menuButton.setAttribute("aria-expanded", String(!isExpanded));
      menuButton.setAttribute("aria-label", isExpanded ? "Open menu" : "Close menu");
      mainMenu.hidden = isExpanded;
    });

    document.getElementById("registerButton").addEventListener("click", () => {
      mainMenu.hidden = true;
      menuButton.setAttribute("aria-expanded", "false");
      menuButton.setAttribute("aria-label", "Open menu");
      registrationForm.reset();
      registrationScanStarted = false;
      registrationCaptureComplete = false;
      registrationPoseIndex = 0;
      registrationPoseFrames = 0;
      registrationCalibrationSamples = [];
      registrationBaselinePose = null;
      registrationScanner.dataset.scanState = "idle";
      registrationScanStatus.textContent = "Start scanning when ready";
      registrationScanIndicator.hidden = true;
      registrationCaptureOverlay.hidden = true;
      updateRegistrationScanButton();
      registrationDialog.showModal();
    });

    document.getElementById("closeRegistrationButton").addEventListener("click", () => registrationDialog.close());
    registrationDialog.addEventListener("close", () => {
      if (registrationScanTransition) {
        registrationScanTransition = false;
        return;
      }
      registrationForm.reset();
      registrationScanStarted = false;
      if (registrationCaptureComplete && stream) {
        registrationCaptureComplete = false;
        registrationCaptureOverlay.hidden = true;
        scanning = true;
        cameraCard.classList.add("detecting");
        setStatus("scanning", "Looking for faces", "Center your face in the frame.");
        buttonText.textContent = "Stop scan";
        scanButton.setAttribute("aria-label", "Stop face scan");
        detectionTimer = window.setInterval(detectFaces, 300);
        detectFaces();
      } else {
        registrationScanner.dataset.scanState = "idle";
        registrationScanStatus.textContent = "Start scanning when ready";
      }
      updateRegistrationScanButton();
    });

    document.getElementById("continueRegistrationButton").addEventListener("click", () => {
      registrationCaptureOverlay.hidden = true;
      registrationDialog.showModal();
      registrationScanStatus.textContent = "Four-way face scan complete";
      registrationScanner.dataset.scanState = "done";
      updateRegistrationScanButton();
    });

    registrationForm.addEventListener("submit", event => {
      event.preventDefault();
      if (!registrationForm.reportValidity()) return;
      if (!registrationCaptureComplete || detectedFaceCount !== 1) {
        showNotice(!registrationCaptureComplete
          ? "Complete the LEFT, RIGHT, DOWN, and UP face scan before continuing."
          : detectedFaceCount === 0
            ? "Keep one face in the camera frame."
            : "Only one face can be in the frame during registration.");
        return;
      }

      registrationDialog.close();
      showNotice("Demo only: no account was created and no information was saved.");
    });

    document.getElementById("activityButton").addEventListener("click", () => {
      mainMenu.hidden = true;
      menuButton.setAttribute("aria-expanded", "false");
      menuButton.setAttribute("aria-label", "Open menu");
      activityDialog.showModal();
    });

    document.getElementById("closeActivityButton").addEventListener("click", () => activityDialog.close());
    activityDialog.addEventListener("click", event => {
      if (event.target === activityDialog) activityDialog.close();
    });

    document.getElementById("helpButton").addEventListener("click", () => {
      showNotice("Camera access requires your permission. Video is not saved. Face detection may not be supported by this browser.");
    });

    window.addEventListener("pagehide", () => {
      stopScan();
      if (detector && typeof detector.close === "function") detector.close();
      detector = null;
    });

    startScan();
