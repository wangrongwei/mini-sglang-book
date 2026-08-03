import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Mini-SGLang 源码解析',
  description: '5000 行代码看懂推理引擎核心——SGLang 最小可读实现深度剖析',
  lang: 'zh-CN',

  base: '/mini-sglang-book/',

  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.png' }],
    ['meta', { name: 'theme-color', content: '#a78bfa' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Mini-SGLang 源码解析' }],
    ['meta', { property: 'og:description', content: '5000 行代码看懂推理引擎核心——SGLang 最小可读实现深度剖析' }],
  ],

  themeConfig: {
    logo: { src: '/logo.png', alt: 'Mini-SGLang' },

    nav: [
      { text: '开始阅读', link: '/chapters/01-positioning' },
      { text: '目录', link: '/contents' },
      { text: 'GitHub', link: 'https://github.com/sgl-project/mini-sglang' },
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '关于本书', link: '/' },
          { text: '完整目录', link: '/contents' },
        ],
      },
      {
        text: '第一部分：为什么是 Mini-SGLang',
        collapsed: false,
        items: [
          { text: '第 1 章　Mini-SGLang 的定位', link: '/chapters/01-positioning' },
          { text: '第 2 章　代码全览', link: '/chapters/02-code-overview' },
        ],
      },
      {
        text: '第二部分：核心数据结构',
        collapsed: false,
        items: [
          { text: '第 3 章　Request 与 Sequence', link: '/chapters/03-request-sequence' },
          { text: '第 4 章　KV Cache 的简化实现', link: '/chapters/04-kv-cache' },
        ],
      },
      {
        text: '第三部分：Prefill 与 Decode',
        collapsed: false,
        items: [
          { text: '第 5 章　Prefill 过程', link: '/chapters/05-prefill' },
          { text: '第 6 章　Decode 过程', link: '/chapters/06-decode' },
          { text: '第 7 章　Continuous Batching 最小实现', link: '/chapters/07-continuous-batching' },
        ],
      },
      {
        text: '第四部分：调度器',
        collapsed: false,
        items: [
          { text: '第 8 章　简化调度器设计', link: '/chapters/08-scheduler' },
          { text: '第 9 章　内存感知调度', link: '/chapters/09-memory-aware' },
        ],
      },
      {
        text: '第五部分：与生产实现对比',
        collapsed: false,
        items: [
          { text: '第 10 章　Mini vs SGLang：RadixAttention', link: '/chapters/10-vs-sglang' },
          { text: '第 11 章　Mini vs vLLM：BlockManager', link: '/chapters/11-vs-vllm' },
          { text: '第 12 章　从 Mini 到 Production', link: '/chapters/12-mini-to-prod' },
        ],
      },
      {
        text: '第六部分：动手实验',
        collapsed: false,
        items: [
          { text: '第 13 章　运行 Mini-SGLang', link: '/chapters/13-run' },
          { text: '第 14 章　扩展 Mini-SGLang', link: '/chapters/14-extend' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '附录 A：代码对照表', link: '/chapters/appendix-a-comparison' },
          { text: '附录 B：关键函数速查', link: '/chapters/appendix-b-functions' },
          { text: '附录 C：名词解释', link: '/chapters/appendix-c-glossary' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/sgl-project/mini-sglang' },
    ],

    footer: {
      message: '基于 MIT 协议发布',
      copyright: 'Copyright © 2025-present',
    },

    search: {
      provider: 'local',
    },
  },

  markdown: {
    lineNumbers: true,
  },
})
