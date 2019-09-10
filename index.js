let net;
const webcamElement = document.getElementById("webcam");
const classifier = knnClassifier.create();

function setupWebcam() {
  return new Promise((resolve, reject) => {
    function successCallback(stream) {
      webcamElement.srcObject = stream;
      webcamElement.addEventListener('loadeddata',  () => resolve(), false);
      webcamElement.play();
    }

    var constraints = { audio: false, video: true };

    function errorCallback(error) {
      console.log("navigator.getUserMedia error: ", error);
    }

    navigator.mediaDevices
      .getUserMedia(constraints)
      .then(successCallback)
      .catch(errorCallback);
  });
}

async function app() {
  console.log("Loading mobilenet..");
  document.getElementById("messages").innerText = `Downloading MobileNet model...`;
  // Load the model.
  net = await mobilenet.load();
  document.getElementById("messages").innerText = ``;
  console.log("Sucessfully loaded model");

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    console.log("Added example of class: " + classId)
    const activation = net.infer(webcamElement, "conv_preds");

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  // When clicking a button, add an example for that class.
  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(2));
  document
    .getElementById("class-d")
    .addEventListener("click", () => addExample(3));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, "conv_preds");
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ["A", "B", "C", "D"];
      document.getElementById("console").innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    await tf.nextFrame();
  }
}

app();
