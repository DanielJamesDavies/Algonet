import fs from "fs";
import { loginDeviceByIp, getEnergyUsage } from "tp-link-tapo-connect";

const smartPlugEmail = "email@email.com"; // Add email used for Smart Plug account here
const smartPlugPassword = "password"; // Add password used for Smart Plug account here
const smartPlugIP = "0.0.0.0"; // Add Smart Plug IP Address Here

const duration = Math.floor(8.5 * 60 * 60 * 1000); // In Milliseconds
const filename = "data/power.json";

const startTime = getCurrentTime("h:m:s");
var data = { startTime: startTime, power: [], time: [] };

console.log("Duration: ", duration / 1000 + "s");

async function run() {
	const device_token = await loginDeviceByIp(smartPlugEmail, smartPlugPassword, smartPlugIP);
	if (!device_token?.deviceIp) return;

	const interval = setInterval(async () => {
		const { current_power } = await getEnergyUsage(device_token);
		console.log(current_power);
		data.power.push(current_power);
		data.time.push(getCurrentTime("h:m:s:ms"));
	}, 500 - 8);

	setTimeout(() => {
		clearInterval(interval);
		fs.writeFileSync(filename, JSON.stringify(data), "utf8");
	}, duration + 100);
}
run();

function getCurrentTime(format) {
	return format
		.split(":")
		.map((formatItem) => {
			switch (formatItem) {
				case "h":
					const hours = new Date().getHours();
					return hours < 10 ? "0" + hours : hours;
					break;
				case "m":
					const minutes = new Date().getMinutes();
					return minutes < 10 ? "0" + minutes : minutes;
					break;
				case "s":
					const seconds = new Date().getSeconds();
					return seconds < 10 ? "0" + seconds : seconds;
					break;
				case "ms":
					const milliseconds = new Date().getMilliseconds();
					return milliseconds < 100 ? (milliseconds < 10 ? "00" + milliseconds : "0" + milliseconds) : milliseconds;
					break;
			}
		})
		.join(":");
}
