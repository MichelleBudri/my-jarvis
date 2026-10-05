// Interface text in the assistant's language: Portuguese or English (the fallback).

const pt = {
  states: {
    booting: "Iniciando",
    sleeping: "Em espera",
    listening: "Ouvindo",
    thinking: "Processando",
    speaking: "Falando",
    offline: "Desconectado",
  },
  hints: {
    sleeping: 'Diga "Hey {name}" ou toque no núcleo',
    listening: "Pode falar",
    offline: "Aguardando o assistente…",
  },
  panels: {
    weather: "Clima",
    system: "Sistema",
    timers: "Timers",
    news: "Notícias de IA",
    latency: "Latência",
  },
  feels: "sensação",
  humidity: "umidade",
  wind: "vento",
  sun: "sol",
  rain: "chuva",
  battery: "bateria",
  charging: "carregando",
  cpu: "cpu",
  memory: "memória",
  noTimers: "Nenhum timer ativo",
  timer: "Timer",
  noNews: "Sem notícias por enquanto",
  hoursAgo: "{n} h",
  justNow: "agora",
  you: "Você",
  voice: "voz",
  footer: "Sistemas locais · nada sai deste Mac",
  tools: {
    get_weather: "consultando o clima",
    get_ai_news: "lendo as notícias",
    get_system_status: "verificando o sistema",
    set_volume: "ajustando o volume",
    open_app: "abrindo o app",
    set_timer: "criando o timer",
    change_timer: "ajustando o timer",
    list_timers: "verificando os timers",
    cancel_timer: "cancelando o timer",
    go_to_sleep: "entrando em espera",
  } as Record<string, string>,
};

type Strings = typeof pt;

const en: Strings = {
  states: {
    booting: "Starting up",
    sleeping: "Standing by",
    listening: "Listening",
    thinking: "Processing",
    speaking: "Speaking",
    offline: "Disconnected",
  },
  hints: {
    sleeping: 'Say "Hey {name}" or tap the core',
    listening: "Go ahead",
    offline: "Waiting for the assistant…",
  },
  panels: {
    weather: "Weather",
    system: "System",
    timers: "Timers",
    news: "AI news",
    latency: "Latency",
  },
  feels: "feels",
  humidity: "humidity",
  wind: "wind",
  sun: "sun",
  rain: "rain",
  battery: "battery",
  charging: "charging",
  cpu: "cpu",
  memory: "memory",
  noTimers: "No active timers",
  timer: "Timer",
  noNews: "No news yet",
  hoursAgo: "{n}h ago",
  justNow: "just now",
  you: "You",
  voice: "voice",
  footer: "Local systems · nothing leaves this Mac",
  tools: {
    get_weather: "checking the weather",
    get_ai_news: "reading the news",
    get_system_status: "checking the system",
    set_volume: "setting the volume",
    open_app: "opening the app",
    set_timer: "setting the timer",
    change_timer: "changing the timer",
    list_timers: "checking the timers",
    cancel_timer: "cancelling the timer",
    go_to_sleep: "standing by",
  },
};

export type { Strings };

export function stringsFor(language: string): Strings {
  return language.toLowerCase().startsWith("pt") ? pt : en;
}

export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ""));
}
