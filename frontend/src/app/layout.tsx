import { Inter } from "next/font/google";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { Toaster } from "sonner";
import "./globals.css";
// TODO  MC8yOmFIVnBZMlhvaklQb3RvVTZNbTVvWlE9PToxMDY5MjIwOA==

const inter = Inter({ subsets: ["latin"] });
// FIXME  MS8yOmFIVnBZMlhvaklQb3RvVTZNbTVvWlE9PToxMDY5MjIwOA==

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="zh-CN"
      suppressHydrationWarning
    >
      <body
        className={inter.className}
        suppressHydrationWarning
      >
        <NuqsAdapter>{children}</NuqsAdapter>
        <Toaster />
      </body>
    </html>
  );
}
