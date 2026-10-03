import { knowledgeBase } from "./data/knowledge.js";

const $ = (id) => document.getElementById(id);
const stages = ["Evre 1", "Evre 2", "Evre 3", "Evre 4"];
const modelConfig = {
  path: "/models/mobilenet.web.onnx",
  inputSize: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225]
};

const state = {
  rulePlan: null,
  ragContext: [],
  answers: {},
  chatHistory: [],
  modelSession: null,
  uploadedFile: null
};

function yes(value) {
  return ["var", "evet", "şüpheli", "supheli", "pozitif"].includes(String(value).toLowerCase());
}

function collectAnswers() {
  const ids = [
    "lokasyon", "doku", "eksuda", "enfeksiyon", "koku", "slough", "nekroz",
    "kizariklik", "isi_artisi", "agri", "ates", "diyabet", "damar_hastaligi"
  ];
  return Object.fromEntries(ids.map((id) => [id, $(id).value]));
}

function buildRulePlan(stageIndex, answers) {
  const stage = stageIndex + 1;
  const infectionSigns = ["enfeksiyon", "koku", "kizariklik", "isi_artisi", "agri", "ates"]
    .filter((key) => yes(answers[key])).length;
  const suspectedInfection = infectionSigns >= 2 || answers.enfeksiyon === "var";
  const tissueBurden = yes(answers.nekroz) || yes(answers.slough);
  const highExudate = answers.eksuda === "fazla";
  const redFlags = [];

  if ([3, 4].includes(stage)) redFlags.push("Evre 3-4 yara ileri klinik değerlendirme gerektirir.");
  if (suspectedInfection) redFlags.push("Enfeksiyon bulguları var; hekim/yara bakım ekibi değerlendirmesi önerilir.");
  if (yes(answers.ates)) redFlags.push("Ateş sistemik enfeksiyon açısından acil değerlendirme gerektirebilir.");
  if (yes(answers.nekroz)) redFlags.push("Nekrotik doku debridman açısından uzman değerlendirmesi gerektirir.");
  if (answers.lokasyon?.toLowerCase() === "topuk" && yes(answers.nekroz)) redFlags.push("Topukta stabil kuru eskar varsa debridman kararı uzmanla verilmelidir.");
  if (yes(answers.diyabet) || yes(answers.damar_hastaligi)) redFlags.push("Diyabet/damar hastalığı iyileşmeyi yavaşlatır; yakın takip gerekir.");

  let ana_urun;
  let yardimci_urun;
  let bakim;

  if (stage === 1) {
    ana_urun = "Barrier film veya koruyucu bariyer krem";
    yardimci_urun = "İnce silikon köpük örtü";
    bakim = ["Basıncı azalt ve düzenli pozisyon değişimi planla", "Cildi temiz, kuru ve travmadan uzak tut", "Kızarıklık alanını günlük izle"];
  } else if (stage === 2) {
    ana_urun = suspectedInfection ? "Gümüş içerikli silikon köpük veya antimikrobiyal örtü" : highExudate ? "Silikon köpük örtü" : "Hidrokolloid veya ince silikon köpük";
    yardimci_urun = suspectedInfection || highExudate ? "Barrier film" : "Nemlendirici bariyer krem";
    bakim = ["Nemli iyileşme ortamını koru", "Sürtünme ve makaslama kuvvetlerini azalt", "Epitel dokuyu travmatik pansumandan koru"];
  } else if (stage === 3) {
    ana_urun = tissueBurden ? "Hidrojel veya hidrofiber, debridman değerlendirmesi ile" : highExudate ? "Aljinat veya yüksek absorban hidrofiber" : suspectedInfection ? "Gümüşlü hidrofiber" : "Hidrofiber veya köpük örtü";
    yardimci_urun = suspectedInfection || tissueBurden ? "Sekonder silikon köpük örtü" : "Barrier film";
    bakim = ["Eksüda miktarına göre pansuman sıklığını ayarla", "Granülasyon dokusunu koru ve maserasyonu izle", "Basınç azaltma ve beslenme değerlendirmesini plana ekle"];
  } else {
    ana_urun = suspectedInfection ? "Gümüşlü hidrofiber veya antimikrobiyal absorban örtü" : tissueBurden ? "Debridman değerlendirmesi sonrası hidrofiber/hidrojel" : "Hidrofiber veya aljinat";
    yardimci_urun = "Sekonder yüksek absorban köpük";
    bakim = ["Derin doku, tünel ve kavite varlığını klinik olarak değerlendir", "Enfeksiyon, nekroz ve eksüdayı yakın takip et", "Basıncı tamamen boşaltacak destek yüzeyi kullan"];
  }

  const risk = redFlags.length || [3, 4].includes(stage) || highExudate ? "yüksek" : "orta";
  const takip = risk === "yüksek" ? "24 saat içinde / günlük" : "48-72 saat";
  return { ana_urun, yardimci_urun, bakim, risk, takip, red_flags: redFlags };
}

function retrieveContext(stageName, answers, topK = 4) {
  const query = `${stageName} ${Object.values(answers).join(" ")}`.toLowerCase();
  const tokens = new Set(query.split(/\s+/).filter(Boolean));
  return knowledgeBase
    .map((item) => {
      const haystack = `${item.title} ${item.tags.join(" ")} ${item.content}`.toLowerCase();
      const score = [...tokens].filter((token) => haystack.includes(token)).length;
      return { ...item, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function setModelResult(title, detail, isError = false) {
  $("modelResult").classList.toggle("error", isError);
  $("modelResult").innerHTML = `<strong>${title}</strong><span>${detail}</span>`;
}

async function ensureModelSession() {
  if (state.modelSession) return state.modelSession;
  if (!window.ort) {
    throw new Error("ONNX Runtime yüklenemedi. İnternet bağlantısını kontrol edin.");
  }
  window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  state.modelSession = await window.ort.InferenceSession.create(modelConfig.path, {
    executionProviders: ["wasm"]
  });
  return state.modelSession;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

async function fileToImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function imageToTensor(file) {
  const image = await fileToImage(file);
  const size = modelConfig.inputSize;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.max(size / image.naturalWidth, size / image.naturalHeight);
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  const dx = (size - width) / 2;
  const dy = (size - height) / 2;
  context.drawImage(image, dx, dy, width, height);

  const pixels = context.getImageData(0, 0, size, size).data;
  const data = new Float32Array(1 * 3 * size * size);
  const plane = size * size;

  for (let i = 0; i < plane; i += 1) {
    const pixelIndex = i * 4;
    data[i] = ((pixels[pixelIndex] / 255) - modelConfig.mean[0]) / modelConfig.std[0];
    data[plane + i] = ((pixels[pixelIndex + 1] / 255) - modelConfig.mean[1]) / modelConfig.std[1];
    data[(2 * plane) + i] = ((pixels[pixelIndex + 2] / 255) - modelConfig.mean[2]) / modelConfig.std[2];
  }

  return new window.ort.Tensor("float32", data, [1, 3, size, size]);
}

async function predictStage() {
  if (!state.uploadedFile) {
    setModelResult("Görüntü bekleniyor", "Tahmin almak için önce yara fotoğrafı yükleyin.", true);
    return;
  }

  $("predictBtn").disabled = true;
  $("predictBtn").textContent = "Model çalışıyor...";
  setModelResult("Model hazırlanıyor", "İlk tahminde model dosyası indirildiği için birkaç saniye sürebilir.");

  try {
    const session = await ensureModelSession();
    const tensor = await imageToTensor(state.uploadedFile);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const outputs = await session.run({ [inputName]: tensor });
    const scores = Array.from(outputs[outputName].data);
    const probabilities = softmax(scores);
    const bestIndex = probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
    const confidence = Math.round(probabilities[bestIndex] * 1000) / 10;

    $("stageSelect").value = String(bestIndex);
    setModelResult(`${stages[bestIndex]} tahmini`, `Güven skoru: %${confidence}. Gerekirse evreyi manuel düzeltebilirsiniz.`);
  } catch (error) {
    setModelResult("Tahmin alınamadı", error.message, true);
  } finally {
    $("predictBtn").disabled = false;
    $("predictBtn").textContent = "Görüntüden Evre Tahmin Et";
  }
}

function renderPlan() {
  const stageIndex = Number($("stageSelect").value);
  const stageName = stages[stageIndex];
  const answers = collectAnswers();
  const rulePlan = buildRulePlan(stageIndex, answers);
  const ragContext = retrieveContext(stageName, answers);
  state.answers = answers;
  state.rulePlan = rulePlan;
  state.ragContext = ragContext;

  $("planCard").innerHTML = `
    <h2>Bakım Önerisi</h2>
    <p><strong>Önerilen ana bakım:</strong> ${rulePlan.ana_urun}</p>
    <p><strong>Destekleyici bakım:</strong> ${rulePlan.yardimci_urun}</p>
    <p><strong>Risk:</strong> <span class="${rulePlan.risk === "yüksek" ? "risk-high" : ""}">${rulePlan.risk}</span> · <strong>Takip:</strong> ${rulePlan.takip}</p>
    <h3>Bakım öncelikleri</h3>
    <ul class="note-list">${rulePlan.bakim.map((item) => `<li>${item}</li>`).join("")}</ul>
    ${rulePlan.red_flags.length ? `<h3>Dikkat edilmesi gerekenler</h3><ul class="note-list">${rulePlan.red_flags.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
  `;

  $("guideCard").innerHTML = `
    <h2>${stageName} İçin Rehber</h2>
    <p>${ragContext[0]?.content || "Bu evre için rehber notu bakım planı oluşturulduktan sonra görüntülenir."}</p>
  `;

  $("sourceList").innerHTML = ragContext.length
    ? ragContext.map((item) => `<div class="source-item"><strong>${item.title}</strong>${item.content}</div>`).join("")
    : `<div class="source-item">Bu vaka için kısa rehber notu bulunamadı.</div>`;
}

function addMessage(role, content, isError = false) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${isError ? "error" : role}`;
  bubble.textContent = content;
  $("chatLog").appendChild(bubble);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

async function askAya() {
  if (!state.rulePlan) {
    addMessage("assistant", "Önce bakım önerisi oluşturun.", true);
    return;
  }
  const question = $("questionInput").value.trim();
  if (!question) return;
  $("questionInput").value = "";
  addMessage("user", question);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        stageName: stages[Number($("stageSelect").value)],
        answers: state.answers,
        rulePlan: state.rulePlan,
        ragContext: state.ragContext,
        history: state.chatHistory
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error));
    addMessage("assistant", data.answer);
    state.chatHistory.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
  } catch (error) {
    addMessage("assistant", `Yanıt alınamadı: ${error.message}`, true);
  }
}

$("imageInput").addEventListener("change", () => {
  const file = $("imageInput").files[0];
  if (!file) return;
  state.uploadedFile = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").parentElement.classList.add("has-image");
  $("dropText").textContent = "Görüntüyü değiştir";
  setModelResult("Görüntü hazır", "Tahmin almak için model butonuna basın.");
});

$("planBtn").addEventListener("click", renderPlan);
$("predictBtn").addEventListener("click", predictStage);
$("chatBtn").addEventListener("click", askAya);
$("questionInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") askAya();
});
