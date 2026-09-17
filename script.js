const policyData = {
  pi05: {
    label: "π₀.₅",
    values: [18.7, 49.3, 58.7, 0, 58.7, 80.0],
  },
  openvla: {
    label: "OpenVLA-OFT",
    values: [4.0, 21.3, 32.0, 0, 25.3, 44.0],
  },
};

const barLabels = ["Teleop 50", "Teleop 100", "Teleop 150", "GS 300", "GS 900", "GS 1500"];

const demos = {
  gpu: {
    src: "assets/videos/gpu-reach.mp4",
    index: "01",
    title: "GPU precise reach",
    copy: "Closed-loop reaching under a changed target-board location.",
  },
  screw: {
    src: "assets/videos/screw-reach.mp4",
    index: "02",
    title: "Screw precise reach",
    copy: "The policy approaches a small interaction region using wrist-camera observations only.",
  },
  ram: {
    src: "assets/videos/ram-reach.mp4",
    index: "03",
    title: "RAM precise reach",
    copy: "The same policy structure reaches a different component after workspace repositioning.",
  },
};

const chart = document.querySelector("#bar-chart");
const chartTitle = document.querySelector("#chart-title");

function renderChart(policy) {
  const data = policyData[policy];
  chart.innerHTML = "";
  chartTitle.textContent = data.label;

  data.values.forEach((value, index) => {
    const item = document.createElement("div");
    item.className = "bar-item";
    item.innerHTML = `
      <span class="bar-value">${value.toFixed(1)}%</span>
      <span class="bar ${index > 2 ? "generated" : "teleop"}" style="height: ${value}%"></span>
      <span class="bar-name">${barLabels[index]}</span>
    `;
    chart.appendChild(item);
  });

  chart.setAttribute("aria-label", `${data.label} average precise reaching success: ${data.values.join(", ")} percent.`);
}

document.querySelectorAll("[data-policy]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-policy]").forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    renderChart(button.dataset.policy);
  });
});

document.querySelectorAll("[data-demo]").forEach((button) => {
  button.addEventListener("click", () => {
    const demo = demos[button.dataset.demo];
    const video = document.querySelector("#demo-video");
    document.querySelectorAll("[data-demo]").forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-selected", String(active));
    });
    video.src = demo.src;
    video.play().catch(() => {});
    document.querySelector("#demo-index").textContent = demo.index;
    document.querySelector("#demo-title").textContent = demo.title;
    document.querySelector("#demo-copy").textContent = demo.copy;
  });
});

const navToggle = document.querySelector(".nav-toggle");
const nav = document.querySelector("#site-nav");
navToggle.addEventListener("click", () => {
  const open = nav.classList.toggle("is-open");
  navToggle.setAttribute("aria-expanded", String(open));
});

nav.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    navToggle.setAttribute("aria-expanded", "false");
  });
});

const progress = document.querySelector("#scroll-progress");
window.addEventListener("scroll", () => {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0}%`;
}, { passive: true });

document.querySelector("#copy-citation").addEventListener("click", async (event) => {
  const text = document.querySelector("#bibtex code").textContent;
  const label = event.currentTarget.querySelector("b");
  try {
    await navigator.clipboard.writeText(text);
    label.textContent = "Copied";
  } catch {
    label.textContent = "Select below";
  }
  setTimeout(() => { label.textContent = "Copy"; }, 1800);
});

if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll("video[autoplay]").forEach((video) => video.pause());
}

renderChart("pi05");
