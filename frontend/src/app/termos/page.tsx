import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Termos de Uso — Top Haws",
  description: "Termos de uso, licença e atribuição do material do curso Top Haws por Ronnald Hawk.",
};

export default function TermsPage() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-8 px-6 py-10 text-slate-100">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.35em] text-teal-200/70">Top Haws</p>
        <h1 className="text-3xl font-semibold text-white">Termos de Uso e Atribuição</h1>
        <p className="text-sm text-slate-300">
          Material do curso produzido por Ronnald Hawk. Estes termos cobrem o uso, redistribuição e atribuição do
          conteúdo.
        </p>
      </header>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">1. Licença e uso permitido</h2>
        <ul className="mt-3 space-y-2 text-sm text-slate-200">
          <li>• Uso permitido apenas a membros ativos e verificados da comunidade Top Haws, conforme a LICENÇA DA COMUNIDADE TOPHAWKS.</li>
          <li>• Membros podem baixar, rodar, copiar, modificar e adaptar o software para criar e operar suas próprias soluções, inclusive comerciais, e disponibilizá-las ao público final, desde que respeitem estas condições.</li>
          <li>• Atribuição mínima: manter referências a Ronnald Hawk e o site{" "}
            <Link className="text-teal-200 hover:text-teal-100" href="https://www.rhawk.pro/">
              rhawk.pro
            </Link>.
          </li>
        </ul>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">2. Marcas e links oficiais</h2>
        <ul className="mt-3 space-y-2 text-sm text-slate-200">
          <li>• Comunidade: Top Haws</li>
          <li>
            • Site:{" "}
            <Link className="text-teal-200 hover:text-teal-100" href="https://www.rhawk.pro/">
              https://www.rhawk.pro/
            </Link>
          </li>
          <li>
            • YouTube:{" "}
            <Link
              className="text-teal-200 hover:text-teal-100"
              href="https://www.youtube.com/channel/UCPiCs9REsEymr43a0ceL_BQ"
            >
              Canal do Ronnald Hawk
            </Link>
          </li>
        </ul>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">3. Monitoramento discreto</h2>
        <p className="mt-3 text-sm text-slate-200">
          Este app envia um sinal leve de uso para rhawk.pro para identificar domínios que hospedam o material. O
          payload inclui origem, nome do app, autor e timestamp, sem dados sensíveis.
        </p>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">4. Distribuição e limitações</h2>
        <ul className="mt-3 space-y-2 text-sm text-slate-200">
          <li>• É proibido comercializar o software ou derivados como produto educacional, curso ou material instrucional concorrente/imitador da Top Haws.</li>
          <li>• É vedado apresentar o software ou derivados como criação original sem atribuir à Top Haws/Ronnald Hawk.</li>
          <li>• Não remover avisos de direitos autorais ou os termos desta licença.</li>
          <li>• Repositórios com o software ou derivados não podem ser públicos; mantenha-os privados ou com acesso restrito a membros.</li>
          <li>• Violação resulta em rescisão imediata desta licença, obrigação de cessar uso e remover o software/derivados de circulação pública, sujeitando o infrator a medidas legais (indenização por perdas e danos, tutela inibitória e demais previstas em lei).</li>
        </ul>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">5. Garantias e responsabilidade</h2>
        <p className="mt-3 text-sm text-slate-200">
          O software é fornecido “no estado em que se encontra”, sem garantias. A Top Haws e autores não se
          responsabilizam por danos decorrentes do uso.
        </p>
      </section>

      <section className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
        <h2 className="text-xl font-semibold text-white">6. Contato</h2>
        <p className="mt-3 text-sm text-slate-200">
          Para dúvidas ou permissões adicionais, acesse{" "}
          <Link className="text-teal-200 hover:text-teal-100" href="https://www.rhawk.pro/">
            rhawk.pro
          </Link>{" "}
          ou entre em contato via comunidade Top Haws.
        </p>
      </section>

      <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Atualizado por Ronnald Hawk</p>
    </div>
  );
}
