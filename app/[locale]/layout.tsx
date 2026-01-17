import { NextIntlClientProvider, useMessages } from 'next-intl';
import { notFound } from 'next/navigation';
import '../globals.css';

export const metadata = {
  title: 'SPEAKEZ - Speech Understanding & Practice Platform',
  description: 'Understand how you speak. Learn how to practice.',
};

export default function RootLayout({
  children,
  params: { locale }
}: {
  children: React.ReactNode,
  params: { locale: string }
}) {
  // Validate that the incoming `locale` parameter is valid
  if (!['en', 'es'].includes(locale)) notFound();

  // Receive messages from the i18n middleware/config
  const messages = useMessages();

  return (
    <html lang={locale}>
      <body>
        <NextIntlClientProvider locale={locale} messages={messages}>
          {children}
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
