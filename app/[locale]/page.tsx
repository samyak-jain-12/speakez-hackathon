"use client";

import Link from 'next/link'
import { Sparkles, Mic, Activity, Map, ArrowRight, Heart, Users, TrendingUp, ShieldCheck } from 'lucide-react'
import { useTranslations } from 'next-intl'
import { motion } from 'framer-motion'

export default function LandingPage() {
  const t = useTranslations()

  // --- Animation Variants ---
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  }

  const item = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
  }

  const slideInLeft = {
    hidden: { opacity: 0, x: -50 },
    whileInView: { opacity: 1, x: 0, transition: { duration: 0.8, ease: "easeOut" } },
    viewport: { once: true }
  }

  const slideInRight = {
    hidden: { opacity: 0, x: 50 },
    whileInView: { opacity: 1, x: 0, transition: { duration: 0.8, ease: "easeOut" } },
    viewport: { once: true }
  }

  const scaleUp = {
    hidden: { opacity: 0, scale: 0.9 },
    whileInView: { opacity: 1, scale: 1, transition: { duration: 0.6 } },
    viewport: { once: true }
  }

  return (
    <main className="min-h-screen relative overflow-hidden bg-white selection:bg-purple-100 selection:text-purple-900">

      {/* --- NAVBAR --- */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6 }}
        className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100/50"
      >
        <div className="container mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <motion.div
              whileHover={{ rotate: 10, scale: 1.1 }}
              className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center text-white font-bold text-xl cursor-default"
            >
              S
            </motion.div>
            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-700 tracking-tight">
              SPEAKEZ
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login" className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors">
              {t('Landing.nav_login')}
            </Link>
            <Link href="/signup">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-5 py-2.5 text-sm font-semibold text-white bg-gray-900 rounded-full shadow-lg shadow-gray-200"
              >
                {t('Landing.nav_signup')}
              </motion.button>
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* --- HERO SECTION --- */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="absolute inset-0 -z-10">
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 10, repeat: Infinity }}
            className="absolute top-0 right-[-10%] w-[800px] h-[800px] bg-purple-100/40 rounded-full blur-[100px]"
          />
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 15, repeat: Infinity, delay: 1 }}
            className="absolute bottom-0 left-[-10%] w-[600px] h-[600px] bg-blue-100/40 rounded-full blur-[100px]"
          />
        </div>

        <div className="container mx-auto px-6 text-center">
          <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="max-w-4xl mx-auto"
          >
            <motion.div variants={item} className="inline-flex items-center justify-center">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-50 border border-purple-100 mb-8">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                </span>
                <span className="text-xs font-semibold text-purple-600 tracking-wide uppercase">AI-Powered Speech Coach</span>
              </div>
            </motion.div>

            <motion.h1 variants={item} className="text-6xl md:text-8xl font-black text-gray-900 tracking-tight leading-[1.1] mb-8">
              {t('Landing.hero_title')} <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600">
                {t('Landing.hero_title_highlight')}
              </span>
            </motion.h1>

            <motion.p variants={item} className="text-xl md:text-2xl text-gray-500 font-light mb-12 max-w-2xl mx-auto leading-relaxed">
              {t('Landing.hero_subtitle')}
            </motion.p>

            <motion.div variants={item} className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20">
              <Link href="/dashboard/practice" passHref>
                <motion.div
                  whileHover={{ scale: 1.05, boxShadow: "0 20px 25px -5px rgba(124, 58, 237, 0.3)" }}
                  whileTap={{ scale: 0.95 }}
                  className="cursor-pointer px-8 py-4 text-lg font-bold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-full transition-all duration-300"
                >
                  {t('Landing.hero_cta')}
                </motion.div>
              </Link>
              <Link href="/login" className="text-gray-500 hover:text-gray-900 font-medium px-6 hover:underline">
                {t('Landing.hero_login')}
              </Link>
            </motion.div>

            {/* Social Proof Stats */}
            <motion.div variants={item} className="grid grid-cols-3 gap-8 border-t border-gray-100 pt-12 max-w-3xl mx-auto">
              <div>
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1, transition: { delay: 0.2 } }}
                  className="text-3xl font-bold text-gray-900 mb-1"
                >
                  10k+
                </motion.div>
                <div className="text-sm text-gray-500 font-medium">{t('Landing.stats_users')}</div>
              </div>
              <div>
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1, transition: { delay: 0.4 } }}
                  className="text-3xl font-bold text-gray-900 mb-1"
                >
                  500k+
                </motion.div>
                <div className="text-sm text-gray-500 font-medium">{t('Landing.stats_sessions')}</div>
              </div>
              <div>
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1, transition: { delay: 0.6 } }}
                  className="text-3xl font-bold text-gray-900 mb-1"
                >
                  4.9/5
                </motion.div>
                <div className="text-sm text-gray-500 font-medium">{t('Landing.stats_rating')}</div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* --- FEATURE 1: REAL TIME --- */}
      <section className="py-24 bg-gray-50/50">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center gap-16">
            <motion.div variants={slideInLeft} initial="hidden" whileInView="whileInView" className="flex-1">
              <div className="w-12 h-12 bg-blue-100 rounded-2xl flex items-center justify-center mb-6 text-blue-600">
                <Activity className="w-6 h-6" />
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-2">{t('Landing.feature_1_title')}</h2>
              <p className="text-sm font-bold text-blue-600 uppercase tracking-widest mb-6">{t('Landing.feature_1_highlight')}</p>
              <p className="text-lg text-gray-600 leading-relaxed mb-8">
                {t('Landing.feature_1_desc')}
              </p>
            </motion.div>

            {/* Abstract Visual: Signal Processing */}
            <motion.div variants={slideInRight} initial="hidden" whileInView="whileInView" className="flex-1 w-full">
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="relative aspect-video bg-white rounded-3xl shadow-2xl border border-gray-100 overflow-hidden flex flex-col p-6"
              >
                {/* Mock UI */}
                <div className="flex items-center justify-between mb-8 border-b border-gray-100 pb-4">
                  <div className="flex gap-2">
                    <div className="w-20 h-2 bg-gray-100 rounded-full animate-pulse"></div>
                    <div className="w-12 h-2 bg-gray-100 rounded-full"></div>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-green-100"></div>
                </div>
                <div className="flex-1 flex gap-2 items-end justify-center px-8">
                  {[40, 60, 30, 80, 50, 90, 40, 70, 45, 60].map((h, i) => (
                    <motion.div
                      key={i}
                      initial={{ height: "10%" }}
                      whileInView={{ height: `${h}%` }}
                      viewport={{ once: false }}
                      transition={{ duration: 1, delay: i * 0.05, ease: "backOut" }}
                      className="w-full bg-blue-500 rounded-t-sm opacity-20"
                    />
                  ))}
                  {/* Active Bar */}
                  <motion.div
                    animate={{ height: ["40%", "80%", "50%"] }}
                    transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse" }}
                    className="absolute bottom-6 left-1/2 w-1 bg-red-500"
                  />
                </div>
                <div className="mt-4 flex justify-between text-xs font-mono text-gray-400">
                  <span>PACE: 145 WPM</span>
                  <span>STRESS: LOW</span>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- FEATURE 2: SCENARIOS --- */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row-reverse items-center gap-16">
            <motion.div variants={slideInRight} initial="hidden" whileInView="whileInView" className="flex-1">
              <div className="w-12 h-12 bg-purple-100 rounded-2xl flex items-center justify-center mb-6 text-purple-600">
                <Users className="w-6 h-6" />
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-2">{t('Landing.feature_2_title')}</h2>
              <p className="text-sm font-bold text-purple-600 uppercase tracking-widest mb-6">{t('Landing.feature_2_highlight')}</p>
              <p className="text-lg text-gray-600 leading-relaxed mb-8">
                {t('Landing.feature_2_desc')}
              </p>
            </motion.div>

            {/* Abstract Visual: Chat Bubble / Simulation */}
            <motion.div variants={slideInLeft} initial="hidden" whileInView="whileInView" className="flex-1 w-full">
              <motion.div
                whileHover={{ rotate: -1 }}
                className="relative aspect-video bg-gray-900 rounded-3xl shadow-2xl overflow-hidden p-8 flex flex-col justify-center"
              >
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                  className="bg-gray-800/50 rounded-2xl p-4 mb-4 self-start max-w-[80%] border border-gray-700"
                >
                  <div className="w-32 h-2 bg-gray-600 rounded-full mb-2"></div>
                  <div className="w-48 h-2 bg-gray-600 rounded-full"></div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 }}
                  className="bg-purple-600 rounded-2xl p-4 mb-4 self-end max-w-[80%] shadow-lg shadow-purple-900/20"
                >
                  <p className="text-white text-sm font-medium">"I can definitely handle that objection by..."</p>
                </motion.div>
                <motion.div
                  initial={{ scale: 0 }}
                  whileInView={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 200, delay: 1 }}
                  className="bg-gray-800/50 rounded-2xl p-4 self-start max-w-[80%] border-l-4 border-yellow-500 bg-yellow-900/10"
                >
                  <p className="text-yellow-400 text-xs font-bold mb-1">CURVEBALL DETECTED</p>
                  <p className="text-gray-300 text-sm">"But what if the budget gets cut in half tomorrow?"</p>
                </motion.div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- FEATURE 3: ANALYTICS --- */}
      <section className="py-24 bg-gray-50/50">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center gap-16">
            <motion.div variants={slideInLeft} initial="hidden" whileInView="whileInView" className="flex-1">
              <div className="w-12 h-12 bg-pink-100 rounded-2xl flex items-center justify-center mb-6 text-pink-600">
                <TrendingUp className="w-6 h-6" />
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-2">{t('Landing.feature_3_title')}</h2>
              <p className="text-sm font-bold text-pink-600 uppercase tracking-widest mb-6">{t('Landing.feature_3_highlight')}</p>
              <p className="text-lg text-gray-600 leading-relaxed mb-8">
                {t('Landing.feature_3_desc')}
              </p>
            </motion.div>

            {/* Abstract Visual: Heatmap / Score */}
            <motion.div variants={scaleUp} initial="hidden" whileInView="whileInView" className="flex-1 w-full">
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="relative aspect-video bg-white rounded-3xl shadow-2xl border border-gray-100 p-8 flex items-center justify-center"
              >
                <div className="relative w-48 h-48">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="96" cy="96" r="88" stroke="#f3f4f6" strokeWidth="12" fill="none" />
                    <motion.circle
                      cx="96" cy="96" r="88" stroke="#ec4899" strokeWidth="12" fill="none"
                      strokeDasharray="552"
                      initial={{ strokeDashoffset: 552 }} // Start empty
                      whileInView={{ strokeDashoffset: 100 }} // Animate to full (mostly)
                      transition={{ duration: 2, ease: "easeOut" }}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.span
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: 1 }}
                      transition={{ delay: 1 }}
                      className="text-5xl font-black text-gray-900"
                    >
                      88
                    </motion.span>
                    <span className="text-xs font-bold text-gray-400 uppercase mt-1">Confidence</span>
                  </div>
                </div>
                <div className="absolute top-8 right-8 flex flex-col gap-2">
                  <motion.div
                    initial={{ x: 20, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="w-24 h-8 bg-green-50 rounded-lg flex items-center px-3 gap-2"
                  >
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-xs font-bold text-green-700">Eye Contact</span>
                  </motion.div>
                  <motion.div
                    initial={{ x: 20, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.7 }}
                    className="w-24 h-8 bg-blue-50 rounded-lg flex items-center px-3 gap-2"
                  >
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    <span className="text-xs font-bold text-blue-700">Volume</span>
                  </motion.div>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- IMPACT SECTION --- */}
      <section className="py-24 bg-gray-900 text-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-20">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              className="text-4xl md:text-5xl font-bold mb-6"
            >
              {t('Landing.impact_title')}
            </motion.h2>
          </div>

          <div className="grid md:grid-cols-3 gap-12">
            {[1, 2, 3].map((i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                whileHover={{ y: -10, backgroundColor: "rgba(31, 41, 55, 0.8)" }}
                className="bg-gray-800/50 p-8 rounded-3xl border border-gray-700 cursor-default"
              >
                <div className="w-12 h-12 bg-gray-700 rounded-xl mb-6 flex items-center justify-center">
                  {i === 1 && <TrendingUp className="w-6 h-6 text-green-400" />}
                  {i === 2 && <ShieldCheck className="w-6 h-6 text-blue-400" />}
                  {i === 3 && <Heart className="w-6 h-6 text-pink-400" />}
                </div>
                {/* @ts-ignore - Dynamic key access */}
                <h3 className="text-xl font-bold mb-4">{t(`Landing.impact_${i}_title`)}</h3>
                {/* @ts-ignore */}
                <p className="text-gray-400 leading-relaxed">{t(`Landing.impact_${i}_text`)}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* --- FINAL CTA --- */}
      <section className="py-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-blue-600/5 -z-10"></div>
        <div className="container mx-auto px-6 text-center">
          <motion.h2
            initial={{ scale: 0.9, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="text-5xl md:text-7xl font-black text-gray-900 mb-8 tracking-tight"
          >
            {t('Landing.cta_final_title')}
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-2xl text-gray-500 font-light mb-12"
          >
            {t('Landing.cta_final_subtitle')}
          </motion.p>
          <Link href="/dashboard/practice" passHref>
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center justify-center px-10 py-5 text-xl font-bold text-white bg-gray-900 rounded-full cursor-pointer shadow-2xl"
            >
              {t('Landing.cta_final_button')}
              <ArrowRight className="ml-3 w-6 h-6" />
            </motion.div>
          </Link>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer className="py-12 border-t border-gray-100 bg-white">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <span className="font-bold text-gray-900">SPEAKEZ</span>
          <div className="text-sm text-gray-500">
            {t('Landing.footer_rights')}
          </div>
          <div className="flex gap-6 text-sm font-medium text-gray-600">
            <a href="#" className="hover:text-gray-900">{t('Landing.footer_privacy')}</a>
            <a href="#" className="hover:text-gray-900">{t('Landing.footer_terms')}</a>
          </div>
        </div>
      </footer>

    </main>
  )
}
