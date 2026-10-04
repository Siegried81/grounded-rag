// UI translations and the language context.
//
// Every user-visible string of the React app lives here, keyed by the same id
// in each UI language, so components never hold display text themselves.
// Strings may use the inline Markdown subset of markdown.jsx (**bold**,
// `code`), rendered with <Inline>, and {name} placeholders filled by t().
// The answer language (what the LLM writes in) is a separate setting sent to
// the API; this module only covers the interface around it.
import { createContext, createElement, useContext, useEffect, useMemo, useState } from "react";

export const UI_LANGUAGES = ["en", "fr"];
export const DEFAULT_UI_LANGUAGE = "en";
// Used when /api/config predates the language fields.
export const FALLBACK_ANSWER_LANGUAGES = ["auto", "en", "fr", "nl"];
const STORAGE_KEY = "grounded-rag.ui-lang";

const en = {
  appTitle: "Grounded RAG assistant",
  headerRule:
    "Answers **only from your documents**, cites every claim as `[S#]`, and **refuses** when nothing relevant is found.",
  openSettings: "Open settings",
  activeConfig: "Active configuration",
  badgeCorpus: "corpus: {value}",
  badgeRetrieval: "retrieval: {value}",
  badgeLlm: "LLM: {provider} · {model}",
  badgeEmbeddings: "embeddings: {value}",
  badgeRefusal: "refusal below {value} cosine",
  badgeNoKey: "no LLM key set",

  languageControls: "Language",
  uiLanguage: "Interface language",
  uiLanguageShort: { en: "EN", fr: "FR" },
  uiLanguageName: { en: "English", fr: "Français" },
  answerIn: "Answer in",
  // Language options of the "Answer in" select: endonyms, so a reader finds
  // their own language whatever the interface language is.
  answerLanguageOption: {
    auto: "Auto (same as sources)",
    en: "English",
    fr: "Français",
    nl: "Nederlands",
  },
  // Language names used inside a sentence ("Answered in French").
  languageName: { en: "English", fr: "French", nl: "Dutch" },
  answeredIn: "Answered in {language}",
  answeredInSourceLanguage: "Answered in the language of the sources",

  settings: "Settings",
  closeSettings: "Close settings",
  corpus: "Corpus",
  documents: "Documents",
  chunks: "Chunks",
  detectedLanguage: "Language (detected): {value}",
  unknown: "unknown",
  pdfPages: " · {n} PDF pages",
  indexedFiles: "Indexed files ({n})",
  retrieval: "Retrieval",
  mode: "Mode",
  noBm25: "No BM25 index for this corpus: re-ingest to enable hybrid mode.",
  modesHelp: "hybrid = embeddings + BM25 keywords fused by rank (RRF). dense = embeddings only.",
  topK: "Passages per answer (top_k):",
  useMmr: "Diversify passages (MMR)",
  thresholdHelp:
    "Refusal threshold: {value} dense cosine (set `SCORE_THRESHOLD` in .env; not editable here because it defines what a refusal means).",
  clearConversation: "Clear conversation",
  howItWorks: "About / How it works",
  limitations: "Limitations",

  emptyTitle: "Ask a question about the selected corpus",
  emptyBody:
    "Each answer cites the passages it uses, which are listed underneath with their scores. If nothing in the documents is relevant enough, the assistant says so instead of guessing.",
  tryThese: "Try one of these",

  question: "Question",
  composerPlaceholder: "Ask a question about this corpus",
  ask: "Ask",
  working: "Working...",

  loading: "Loading...",
  noCorpus: "No indexed corpus found. Run `python cli.py ingest --corpus NAME` first.",
  apiUnreachable: "Could not reach the API.",
  apiStartHint: "{error}. Start it with `uvicorn api.main:app --port 8002` and reload.",
  requestFailed: "The request failed.",
  requestFailedHint: "{error}. Is the API running (`uvicorn api.main:app --port 8002`)?",

  thinking: "Retrieving passages and drafting a cited answer...",
  refusalTitle: "No answer: not enough evidence in this corpus.",
  closestPassages: "Closest passages found (below threshold)",
  sources: "Sources",
  scoreNote:
    "Score = cosine similarity to the question (1.0 = identical). In hybrid mode a passage found only by keywords shows its BM25 score instead.",
  sourceLabel: "Source S{sid}: {location}",
  pageShort: "p.{n}",
  scoreTitle: "Cosine similarity to the question (BM25 score if found by keywords only)",
  cited: "cited",
  showSource: "Show source S{n}",
  badCitation: "This citation points to no retrieved source",

  verification: "Verification",
  verified: "✓ verified",
  issuesFound: "! issues found",
  grounding: "grounding",
  invalidCitations: "Invalid citations (no such source): {list}",
  uncitedClaims: "Claims without a citation:",

  timings: "Timings",
  retrievalTime: "retrieval {s}s",
  answerTime: "answer {s}s",
};

const fr = {
  appTitle: "Assistant RAG ancré",
  headerRule:
    "Répond **uniquement à partir de vos documents**, cite chaque affirmation en `[S#]` et **refuse** de répondre quand rien de pertinent n'est trouvé.",
  openSettings: "Ouvrir les paramètres",
  activeConfig: "Configuration active",
  badgeCorpus: "corpus : {value}",
  badgeRetrieval: "recherche : {value}",
  badgeLlm: "LLM : {provider} · {model}",
  badgeEmbeddings: "embeddings : {value}",
  badgeRefusal: "refus sous {value} cosinus",
  badgeNoKey: "aucune clé LLM configurée",

  languageControls: "Langue",
  uiLanguage: "Langue de l'interface",
  uiLanguageShort: { en: "EN", fr: "FR" },
  uiLanguageName: { en: "English", fr: "Français" },
  answerIn: "Répondre en",
  answerLanguageOption: {
    auto: "Auto (comme les sources)",
    en: "English",
    fr: "Français",
    nl: "Nederlands",
  },
  languageName: { en: "anglais", fr: "français", nl: "néerlandais" },
  answeredIn: "Réponse en {language}",
  answeredInSourceLanguage: "Réponse dans la langue des sources",

  settings: "Paramètres",
  closeSettings: "Fermer les paramètres",
  corpus: "Corpus",
  documents: "Documents",
  chunks: "Passages indexés",
  detectedLanguage: "Langue (détectée) : {value}",
  unknown: "inconnue",
  pdfPages: " · {n} pages PDF",
  indexedFiles: "Fichiers indexés ({n})",
  retrieval: "Recherche",
  mode: "Mode",
  noBm25: "Pas d'index BM25 pour ce corpus : réindexez-le pour activer le mode hybride.",
  modesHelp:
    "hybrid = embeddings + mots-clés BM25 fusionnés par rang (RRF). dense = embeddings uniquement.",
  topK: "Passages par réponse (top_k) :",
  useMmr: "Diversifier les passages (MMR)",
  thresholdHelp:
    "Seuil de refus : {value} en cosinus dense (réglez `SCORE_THRESHOLD` dans .env ; non modifiable ici car il définit ce qu'est un refus).",
  clearConversation: "Effacer la conversation",
  howItWorks: "À propos / Fonctionnement",
  limitations: "Limites",

  emptyTitle: "Posez une question sur le corpus sélectionné",
  emptyBody:
    "Chaque réponse cite les passages qu'elle utilise, listés en dessous avec leur score. Si rien dans les documents n'est assez pertinent, l'assistant le dit au lieu de deviner.",
  tryThese: "Essayez l'une de ces questions",

  question: "Question",
  composerPlaceholder: "Posez une question sur ce corpus",
  ask: "Envoyer",
  working: "En cours...",

  loading: "Chargement...",
  noCorpus: "Aucun corpus indexé. Lancez d'abord `python cli.py ingest --corpus NAME`.",
  apiUnreachable: "Impossible de joindre l'API.",
  apiStartHint: "{error}. Démarrez-la avec `uvicorn api.main:app --port 8002` puis rechargez la page.",
  requestFailed: "La requête a échoué.",
  requestFailedHint: "{error}. L'API est-elle démarrée (`uvicorn api.main:app --port 8002`) ?",

  thinking: "Recherche des passages et rédaction d'une réponse citée...",
  refusalTitle: "Pas de réponse : pas assez d'éléments dans ce corpus.",
  closestPassages: "Passages les plus proches (sous le seuil)",
  sources: "Sources",
  scoreNote:
    "Score = similarité cosinus avec la question (1,0 = identique). En mode hybride, un passage trouvé uniquement par mots-clés affiche son score BM25.",
  sourceLabel: "Source S{sid} : {location}",
  pageShort: "p. {n}",
  scoreTitle: "Similarité cosinus avec la question (score BM25 si trouvé uniquement par mots-clés)",
  cited: "cité",
  showSource: "Afficher la source S{n}",
  badCitation: "Cette citation ne correspond à aucune source récupérée",

  verification: "Vérification",
  verified: "✓ vérifié",
  issuesFound: "! problèmes détectés",
  grounding: "ancrage",
  invalidCitations: "Citations invalides (source inexistante) : {list}",
  uncitedClaims: "Affirmations sans citation :",

  timings: "Durées",
  retrievalTime: "recherche {s} s",
  answerTime: "réponse {s} s",
};

const DICTIONARIES = { en, fr };

function readStoredLanguage() {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return UI_LANGUAGES.includes(v) ? v : DEFAULT_UI_LANGUAGE;
  } catch {
    return DEFAULT_UI_LANGUAGE; // storage blocked (private mode, sandboxed iframe)
  }
}

/** Translate `key` for `lang`, falling back to English, then to the key itself
 * so a missing entry shows up visibly instead of rendering nothing. Nested
 * keys use a dot ("languageName.fr"). */
export function translate(lang, key, vars) {
  const lookup = (dict) => key.split(".").reduce((o, k) => (o == null ? o : o[k]), dict);
  let s = lookup(DICTIONARIES[lang]);
  if (s == null) s = lookup(en);
  if (s == null) return key;
  if (!vars) return s;
  return s.replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name]) : m));
}

const I18nContext = createContext(null);

/** Holds the UI language, persists it, and keeps <html lang> in sync so
 * screen readers pronounce the interface in the right language. */
export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(readStoredLanguage);

  useEffect(() => {
    document.documentElement.lang = lang;
    try {
      window.localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* not persisted: the choice still applies for this session */
    }
  }, [lang]);

  const value = useMemo(
    () => ({
      lang,
      setLang: (l) => UI_LANGUAGES.includes(l) && setLangState(l),
      t: (key, vars) => translate(lang, key, vars),
    }),
    [lang]
  );
  return createElement(I18nContext.Provider, { value }, children);
}

/** Returns { t, lang, setLang } for the current UI language. */
export function useT() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useT() must be used inside <I18nProvider>");
  return ctx;
}
