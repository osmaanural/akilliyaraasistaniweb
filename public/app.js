import { knowledgeBase } from "./data/knowledge.js";

const $ = (id) => document.getElementById(id);
const stages = ["Evre 1", "Evre 2", "Evre 3", "Evre 4"];
const modelConfig = {
  path: "./models/mobilenet.web.onnx",
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
  uploadedFile: null,
  predictedStageIndex: null,
  predictionConfidence: null,
  publicUploadedFile: null,
  publicStageIndex: null,
  publicConfidence: null,
  publicChatHistory: [],
  publicSummary: null
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

function setPublicModelResult(title, detail, isError = false) {
  $("publicModelResult").classList.toggle("error", isError);
  $("publicModelResult").innerHTML = `<strong>${title}</strong><span>${detail}</span>`;
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

async function runImagePrediction(file) {
  const session = await ensureModelSession();
  const tensor = await imageToTensor(file);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const outputs = await session.run({ [inputName]: tensor });
  const scores = Array.from(outputs[outputName].data);
  const probabilities = softmax(scores);
  const bestIndex = probabilities.reduce((best, value, index) => value > probabilities[best] ? index : best, 0);
  const confidence = Math.round(probabilities[bestIndex] * 1000) / 10;
  return { bestIndex, confidence };
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
    const { bestIndex, confidence } = await runImagePrediction(state.uploadedFile);

    $("stageSelect").value = String(bestIndex);
    state.predictedStageIndex = bestIndex;
    state.predictionConfidence = confidence;
    const caution = confidence < 65
      ? " Güven skoru düşük; görüntü ve bulgular sağlık profesyoneli tarafından ayrıca değerlendirilmelidir."
      : " Bulgularla birlikte değerlendirilmelidir.";
    setModelResult(`Tahmin Edilen Evre: ${stages[bestIndex]}`, `Güven skoru: %${confidence}.${caution}`);
  } catch (error) {
    setModelResult("Tahmin alınamadı", error.message, true);
  } finally {
    $("predictBtn").disabled = false;
    $("predictBtn").textContent = "Görüntüden Evre Tahmin Et";
  }
}

async function predictPublicStage() {
  if (!state.publicUploadedFile) {
    setPublicModelResult("Fotoğraf bekleniyor", "Ön bilgi almak için önce yara fotoğrafı yükleyin.", true);
    return;
  }

  $("publicPredictBtn").disabled = true;
  $("publicPredictBtn").textContent = "Fotoğraf inceleniyor...";
  setPublicModelResult("Hazırlanıyor", "İlk kullanımda model dosyası indirildiği için birkaç saniye sürebilir.");

  try {
    const { bestIndex, confidence } = await runImagePrediction(state.publicUploadedFile);
    state.publicStageIndex = bestIndex;
    state.publicConfidence = confidence;
    const simpleStage = `Evre ${bestIndex + 1}`;
    const warning = confidence < 65
      ? " Fotoğraf sonucu çok net değil; bir sağlık çalışanının görmesi daha doğru olur."
      : " Bu sonuç kesin karar değildir.";
    setPublicModelResult(`Tahmin Edilen Evre: ${simpleStage}`, `Tahmin netliği: %${confidence}.${warning}`);
  } catch (error) {
    setPublicModelResult("Ön bilgi alınamadı", error.message, true);
  } finally {
    $("publicPredictBtn").disabled = false;
    $("publicPredictBtn").textContent = "Fotoğraftan Ön Bilgi Al";
  }
}

function publicYes(id) {
  return ["evet", "fazla"].includes(String($(id).value).toLowerCase());
}

function renderPublicSummary() {
  const warnings = [];
  const recommendations = [
    "Yara üzerine baskı gelmesini azaltın.",
    "Yarayı temiz ve kuru tutmaya çalışın.",
    "Yarayı kendi başınıza kesmeyin, kazımayın veya derinlemesine temizlemeye çalışmayın.",
  ];

  if (publicYes("publicSmell")) warnings.push("Kötü koku fark ediyorsanız sağlık kuruluşuna başvurmanız önerilir.");
  if (publicYes("publicDark")) warnings.push("Siyah veya koyu alan varsa yara daha dikkatli değerlendirilmelidir.");
  if (publicYes("publicRedness")) warnings.push("Çevrede kızarıklık veya sıcaklık artışı varsa iltihap belirtisi olabilir.");
  if (publicYes("publicPainFever")) warnings.push("Ateş veya artan ağrı varsa gecikmeden yardım alınmalıdır.");
  if ($("publicFluid").value === "fazla") warnings.push("Fazla sıvı gelmesi yaranın yakından izlenmesini gerektirir.");

  const location = $("publicLocation").value.trim() || "belirtilmeyen bölge";
  const stageText = state.publicStageIndex === null
    ? "Fotoğraf sonucu alınmadı."
    : `Fotoğrafa göre yara Evre ${state.publicStageIndex + 1} görünümüne benzer olabilir. Tahmin netliği: %${state.publicConfidence}.`;

  state.publicSummary = {
    location,
    stageText,
    warnings,
    recommendations,
    answers: {
      "yaranın olduğu yer": location,
      "yaradan sıvı geliyor mu": $("publicFluid").value,
      "kötü koku var mı": $("publicSmell").value,
      "siyah veya koyu alan var mı": $("publicDark").value,
      "çevresi kızarık ya da sıcak mı": $("publicRedness").value,
      "ateş veya artan ağrı var mı": $("publicPainFever").value,
    }
  };

  $("publicSummaryCard").innerHTML = `
    <h2>Sade Özet</h2>
    <p><strong>Yara yeri:</strong> ${location}</p>
    <p><strong>Fotoğraf bilgisi:</strong> ${stageText}</p>
    <h3>Dikkat etmeniz gerekenler</h3>
    <ul class="note-list">${(warnings.length ? warnings : ["Şu an belirgin acil uyarı işaretlenmedi. Yine de yara takip edilmeli ve kötüleşirse yardım alınmalıdır."]).map((item) => `<li>${item}</li>`).join("")}</ul>
    <h3>Genel öneriler</h3>
    <ul class="note-list">${recommendations.map((item) => `<li>${item}</li>`).join("")}</ul>
    <p class="subtle">Bu özet tanı veya tedavi değildir. Yara derinse, kötüleşiyorsa ya da emin değilseniz sağlık profesyoneline danışın.</p>
`;
}

function selectAudience(mode) {
  $("professionalApp").classList.toggle("is-hidden", mode !== "professional");
  $("publicApp").classList.toggle("is-hidden", mode !== "public");
  document.querySelectorAll(".audience-card").forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.mode === mode);
  });
  const target = mode === "professional" ? $("professionalApp") : $("publicApp");
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPlan() {
  if (state.predictedStageIndex === null || $("stageSelect").value === "") {
    setModelResult("Evre tahmini gerekli", "Bakım önerisi oluşturmak için önce yara fotoğrafından evre tahmini alın.", true);
    $("planCard").innerHTML = `<p class="empty">Önce görüntüden evre tahmini alınmalıdır.</p>`;
    $("guideCard").innerHTML = `<p class="empty">Tahmin sonrası ilgili rehber özeti görüntülenecek.</p>`;
    return;
  }

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
    <p><strong>Model tahmini:</strong> ${stageName} · <strong>Güven:</strong> %${state.predictionConfidence}</p>
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

function addPublicMessage(role, content, isError = false) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${isError ? "error" : role}`;
  bubble.textContent = content;
  $("publicChatLog").appendChild(bubble);
  $("publicChatLog").scrollTop = $("publicChatLog").scrollHeight;
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

async function askPublicAya() {
  const question = $("publicQuestionInput").value.trim();
  if (!question) return;
  $("publicQuestionInput").value = "";
  addPublicMessage("user", question);

  const location = $("publicLocation").value.trim() || "belirtilmedi";
  const publicAnswers = {
    "yaranın olduğu yer": location,
    "yaradan sıvı geliyor mu": $("publicFluid").value,
    "kötü koku var mı": $("publicSmell").value,
    "siyah veya koyu alan var mı": $("publicDark").value,
    "çevresi kızarık ya da sıcak mı": $("publicRedness").value,
    "ateş veya artan ağrı var mı": $("publicPainFever").value,
  };
  const publicStageText = state.publicStageIndex === null
    ? "fotoğraf tahmini alınmadı"
    : `Evre ${state.publicStageIndex + 1} görünümüne benzer, tahmin netliği %${state.publicConfidence}`;
  const publicWarnings = state.publicSummary?.warnings || [];

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audience: "public",
        question,
        publicStageText,
        stageName: state.publicStageIndex === null ? "belirtilmedi" : stages[state.publicStageIndex],
        answers: publicAnswers,
        rulePlan: {
          summary: state.publicSummary || null,
          warningSigns: publicWarnings,
        },
        ragContext: [
          {
            title: "Sade yara bakım uyarıları",
            content: "Kötü koku, ateş, artan kızarıklık/sıcaklık, fazla sıvı, siyah veya koyu alan, artan ağrı ve yaranın büyümesi durumunda sağlık kuruluşuna başvurulmalıdır."
          }
        ],
        history: state.publicChatHistory
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error));
    addPublicMessage("assistant", data.answer);
    state.publicChatHistory.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
  } catch (error) {
    addPublicMessage("assistant", `Yanıt alınamadı: ${error.message}`, true);
  }
}

$("imageInput").addEventListener("change", () => {
  const file = $("imageInput").files[0];
  if (!file) return;
  state.uploadedFile = file;
  state.predictedStageIndex = null;
  state.predictionConfidence = null;
  $("stageSelect").value = "";
  $("preview").src = URL.createObjectURL(file);
  $("preview").parentElement.classList.add("has-image");
  $("dropText").textContent = "Görüntüyü değiştir";
  setModelResult("Görüntü hazır", "Tahmin almak için model butonuna basın.");
});

document.querySelectorAll(".audience-card").forEach((button) => {
  button.addEventListener("click", () => selectAudience(button.dataset.mode));
});

$("publicImageInput").addEventListener("change", () => {
  const file = $("publicImageInput").files[0];
  if (!file) return;
  state.publicUploadedFile = file;
  state.publicStageIndex = null;
  state.publicConfidence = null;
  $("publicPreview").src = URL.createObjectURL(file);
  $("publicPreview").parentElement.classList.add("has-image");
  $("publicDropText").textContent = "Görüntüyü değiştir";
  setPublicModelResult("Fotoğraf hazır", "Ön bilgi almak için butona basın.");
});

$("planBtn").addEventListener("click", renderPlan);
$("predictBtn").addEventListener("click", predictStage);
$("publicPredictBtn").addEventListener("click", predictPublicStage);
$("publicPlanBtn").addEventListener("click", renderPublicSummary);
$("publicChatBtn").addEventListener("click", askPublicAya);
$("chatBtn").addEventListener("click", askAya);
$("questionInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") askAya();
});
$("publicQuestionInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") askPublicAya();
});
