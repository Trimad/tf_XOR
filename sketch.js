let model;
let resolution = 32;
let cols, rows;
let xs;
const train_xs = tf.tensor2d([
	[0, 0], [1, 0], [0, 1], [1, 1]
]);
const train_ys = tf.tensor2d([
	[0], [1], [1], [0]
]);
function setup() {
	createCanvas(windowHeight, windowHeight);
	noStroke();
	cols = (width / resolution)-1;
	rows = (height / resolution)-1;
	//create the input data
	let inputs = [];
	for (let i = 0; i < cols; i++) {
		for (let j = 0; j < rows; j++) {
			let x1 = i / cols;
			let x2 = j / rows;
			inputs.push([x1, x2]);
		}
	}
	xs = tf.tensor2d(inputs);
	model = tf.sequential();

	let hidden = tf.layers.dense(
		{
			inputShape: [2],
			units: 4,
			activation: 'sigmoid'
		}
	);

	let output = tf.layers.dense(
		{
			units: 1,
			activation: 'sigmoid'
		}
	)

	model.add(hidden);
	model.add(output);
	model.compile(
		{
			optimizer: tf.train.adam(0.1),
			//loss: tf.losses.meanSquaredError
			loss: 'meanSquaredError'
		}
	);
}
async function trainModel() {
	tf.tidy(() => {
		const config = {
			epochs: 2,
			shuffle: true
		}
		model.fit(train_xs, train_ys, config);
	});

	//return await tf.nextFrame();

}
function draw() {

	trainModel();

	//get the predictions
	tf.tidy(() => {
		let ys = model.predict(xs);
		let y_values = ys.dataSync();

		//draw the results
		let index = 0;
		for (let i = 0; i <= cols; i++) {
			for (let j = 0; j <= rows; j++) {
				let br = y_values[index] * 255;

				fill(br);
				rect(i * resolution, j * resolution, resolution, resolution);
				fill(255 - br);
				textSize(8);
				textAlign(CENTER, CENTER);
				text(nf(y_values[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2);
				index++;
			}
		}
	});

	if (frameCount % 10 === 0) {
		document.title = "FPS: " + Math.round(frameRate()) + " Tensors: " + tf.memory().numTensors;
		//console.log(a + ", " + b + ", " + c);
	}

}