import type { Metadata } from "next";
import { Sora } from "next/font/google";
import "./globals.css";
import { GenesisUIProvider } from "@/state/useGenesisUI";
import { AuthProvider } from "@/state/useAuth";
import { AppChrome } from "@/components/layout/AppChrome";

const sora = Sora({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-sans",
});
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "");

export const metadata: Metadata = {
  title: "Commandix AI",
  description: "Console de testes de agentes da Commandix AI",
  applicationName: "Commandix AI",
  icons: {
    icon: "/commandix-logo.png",
    shortcut: "/commandix-logo.png",
    apple: "/commandix-logo.png",
  },
  authors: [{ name: "Ronnald Hawk", url: "https://www.rhawk.pro/" }],
  creator: "Ronnald Hawk",
  publisher: "Ronnald Hawk",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "Commandix AI",
    description: "Console de testes de agentes da Commandix AI",
    url: siteUrl,
    siteName: "Commandix AI",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" className="bg-[#090e1a]">
      <head>
        <meta name="author" content="Ronnald Hawk" />
        <meta name="application-name" content="Commandix AI" />
        <meta name="copyright" content="Ronnald Hawk" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Person",
              name: "Ronnald Hawk",
              url: "https://www.rhawk.pro/",
              sameAs: [
                "https://www.youtube.com/channel/UCPiCs9REsEymr43a0ceL_BQ",
              ],
              affiliation: {
                "@type": "Organization",
                name: "Commandix AI",
                url: "https://www.rhawk.pro/",
              },
            }),
          }}
        />
        {/* Marca discreta para rastreio de autoria no curso */}
      </head>
      <body
        className={`${sora.variable} antialiased text-[#dfdecf]`}
        suppressHydrationWarning
      >
        <AuthProvider>
          <GenesisUIProvider>
            <AppChrome>{children}</AppChrome>
          </GenesisUIProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
