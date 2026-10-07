const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

function buildSystemPrompt(payload) {
  const { stageName, answers, rulePlan, ragContext } = payload;
  if (payload.audience === "public") {
    return `
/no_think
Sen Akıllı Yara Asistanı isimli yara bakım destek asistanısın.
Kullanıcı sağlık çalışanı değil. Tıbbi terim kullanmadan, sade ve anlaşılır Türkçe ile cevap ver.
Tanı koyma, reçete yazma, kesin tedavi iddiasında bulunma.
Antibiyotik, pansuman ürünü veya ilaç ismi önermekten kaçın; bu kararın sağlık profesyoneli tarafından verilmesi gerektiğini söyle.
Kısa cevap ver. Gerekirse 3-5 maddelik liste kullan.
"Eksüda" yerine "yaradan gelen sıvı", "nekroz" yerine "siyah/koyu cansız görünümlü alan", "slough" yerine "sarımsı yumuşak tabaka" gibi sade ifadeler kullan.
Ateş, kötü koku, hızla artan kızarıklık/sıcaklık, yoğun sıvı, siyah/koyu alan, artan ağrı, hızlı büyüme veya derin yara varsa sağlık kuruluşuna başvurmayı net söyle.

Fotoğraf ön bilgisi: ${payload.publicStageText || stageName || "belirtilmedi"}

Kullanıcının işaretlediği basit bulgular:
${JSON.stringify(answers || {}, null, 2)}

Kural/özet bağlamı:
${JSON.stringify(rulePlan || {}, null, 2)}

Rehber bağlamı:
${JSON.stringify(ragContext || [], null, 2)}
`.trim();
  }

  return `
/no_think
Sen Akıllı Yara Asistanı isimli yara bakım destek asistanısın.
Tanı koyma, reçete yazma ve kesin tedavi iddiasında bulunma.
Yalnızca verilen yara evresi, kullanıcı bulguları, kural tabanlı çıktı ve rehber bağlamına dayan.
İç düşünce, analiz adımları veya İngilizce reasoning yazma; sadece nihai Türkçe cevabı ver.
Cevaplarını kısa, anlaşılır ve sağlık çalışanına uygun yaz.
Eksik bilgi varsa önce soru sor.
Ateş, kötü koku, hızla artan kızarıklık, ısı artışı, yoğun akıntı, nekroz veya Evre 3-4 gibi risklerde sağlık profesyoneli değerlendirmesini açıkça vurgula.

Yara evresi: ${stageName}

Kullanıcı bulguları:
${JSON.stringify(answers, null, 2)}

Kural tabanlı çıktı:
${JSON.stringify(rulePlan, null, 2)}

Rehber bağlamı:
${JSON.stringify(ragContext, null, 2)}
`.trim();
}

function extractAnswer(data) {
  const choice = data?.choices?.[0];
  const message = choice?.message || {};
  let answer = message.content;

  if (Array.isArray(answer)) {
    answer = answer
      .map((part) => (typeof part === "object" ? part.text || JSON.stringify(part) : String(part)))
      .join("\n");
  }

  if (typeof answer === "string" && answer.trim()) {
    return answer.trim();
  }

  if (choice?.finish_reason === "length" && message.reasoning) {
    throw new Error("Model cevap yerine düşünme tokenı üretti. Modeli değiştirip tekrar deneyin.");
  }

  throw new Error(`OpenRouter yanıtında cevap metni bulunamadı. finish_reason=${choice?.finish_reason || "bilinmiyor"}`);
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL || "qwen/qwen3.7-flash";

  if (!apiKey) {
    return res.status(500).json({ error: "OPENROUTER_API_KEY tanımlı değil." });
  }

  const { question, history = [] } = req.body || {};
  if (!question || !String(question).trim()) {
    return res.status(400).json({ error: "Soru boş olamaz." });
  }

  const messages = [
    { role: "system", content: buildSystemPrompt(req.body) },
    ...history.slice(-6),
    { role: "user", content: String(question).trim() }
  ];

  try {
    const response = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": req.headers.origin || "https://aya.local",
        "X-Title": "Akilli Yara Asistani"
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.2,
        max_tokens: 900,
        reasoning: {
          max_tokens: 64,
          exclude: true
        }
      })
    });

    const data = await response.json();
    if (!response.ok) {
      return res.status(response.status).json({ error: data });
    }

    return res.status(200).json({
      answer: extractAnswer(data),
      model: data.model,
      usage: data.usage || null
    });
  } catch (error) {
    return res.status(500).json({ error: error.message || "OpenRouter isteği başarısız." });
  }
}
