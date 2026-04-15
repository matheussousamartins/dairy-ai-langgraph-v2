import type { Metadata } from "next";
import { Open_Sans, Roboto_Condensed } from "next/font/google";
import "./globals.css";
import { GenesisUIProvider } from "@/state/useGenesisUI";
import { AuthProvider } from "@/state/useAuth";
import { AppChrome } from "@/components/layout/AppChrome";

const openSans = Open_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans" });
const robotoCondensed = Roboto_Condensed({ subsets: ["latin"], weight: ["300", "400", "700"], variable: "--font-condensed" });
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "");

export const metadata: Metadata = {
  title: "Commandix Tech",
  description: "Shell inicial do agente TopHawks",
  applicationName: "Top Haws",
  authors: [{ name: "Ronnald Hawk", url: "https://www.rhawk.pro/" }],
  creator: "Ronnald Hawk",
  publisher: "Ronnald Hawk",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "Commandix Tech",
    description: "Shell inicial do agente TopHawks",
    url: siteUrl,
    siteName: "Top Haws",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" className="bg-[#090e1a]">
      <head>
        <meta name="author" content="Ronnald Hawk" />
        <meta name="application-name" content="Top Haws" />
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
                name: "Top Haws",
                url: "https://www.rhawk.pro/",
              },
            }),
          }}
        />
        {/* Marca discreta para rastreio de autoria no curso */}
      </head>
      <body
        className={`${openSans.variable} ${robotoCondensed.variable} antialiased text-[#dfdecf]`}
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
