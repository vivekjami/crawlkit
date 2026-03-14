"""
crawlkit token comparison benchmark

Measures real token savings from HTML → clean text extraction.

Methodology
-----------
Dataset:
    100,000 real articles from HuggingFace cc_news dataset.

HTML generation:
    Articles are wrapped in realistic HTML that mirrors what a scraper
    actually receives from a news site: inline tracking scripts,
    expanded navigation with ~30 links, a recommendation sidebar,
    a cookie consent banner, a newsletter signup form, and a full
    multi-column footer. This is the overhead that real pipelines pay.

    Signal (extracted by crawlkit): <main>, <article>, <h1..h6>, <p>, <li>
    Noise (stripped by crawlkit):   <nav>, <header>, <footer>, <aside>,
                                     <script>, <style>, <noscript>, <form>

Extraction comparison:
    Python baseline: selectolax HTML parser (serial, GIL-bound)
    Rust: crawlkit batch_extract_clean_text (parallel via Rayon)

Token counting:
    Real tokenizer using OpenAI tiktoken cl100k_base.

Outputs:
    - Raw HTML tokens vs clean text tokens per page
    - Token reduction %
    - LLM cost at 100k pages/month
    - Extraction speed: Python vs Rust

Run:
    pip install datasets selectolax tiktoken
    maturin develop --release
    python benchmarks/token_comparison.py
"""

import time
from statistics import mean, median, stdev

import tiktoken
from crawlkit import batch_extract_clean_text
from datasets import load_dataset
from selectolax.parser import HTMLParser

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

N_PAGES = 100_000
RUNS = 5

LLM_PRICES = {
    "GPT-4o": 2.50,
    "Claude Sonnet": 3.00,
}

ENC = tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────
# HTML wrapper
#
# Mirrors a real news site scrape:
#   - Inline analytics and tracking scripts  (~800 tokens overhead)
#   - Navigation with ~30 links              (~150 tokens overhead)
#   - Cookie consent banner                  (~120 tokens overhead)
#   - Recommendation sidebar                 (~200 tokens overhead)
#   - Newsletter signup form                 (~100 tokens overhead)
#   - Multi-column footer with ~40 links     (~200 tokens overhead)
#
# Total noise overhead: ~1,500–2,000 tokens per page (conservative).
# Real e-commerce and SaaS pages run 3,000–5,000 tokens of noise.
# ─────────────────────────────────────────────


def wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    body{{font-family:Arial,sans-serif;margin:0;padding:0;color:#1a1a1a}}
    .site-header{{background:#fff;border-bottom:1px solid #e0e0e0;position:sticky;top:0;z-index:100}}
    .nav-wrapper{{max-width:1200px;margin:0 auto;display:flex;align-items:center;padding:0 24px;height:60px}}
    .site-logo{{font-size:22px;font-weight:700;text-decoration:none;color:#c00;margin-right:32px}}
    .nav-primary{{display:flex;list-style:none;margin:0;padding:0;gap:4px}}
    .nav-primary li a{{padding:8px 12px;font-size:14px;font-weight:500;color:#1a1a1a;text-decoration:none;white-space:nowrap}}
    .nav-primary li a:hover{{color:#c00}}
    .layout{{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:1fr 300px;gap:40px;padding:32px 24px}}
    .article-body p{{font-size:17px;line-height:1.75;margin-bottom:20px}}
    .breaking-bar{{background:#c00;color:#fff;padding:6px 24px;font-size:13px;font-weight:600}}
    .sidebar-widget{{border:1px solid #e0e0e0;border-radius:4px;padding:16px;margin-bottom:24px}}
    .sidebar-widget h3{{font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin:0 0 12px}}
    .trending-list{{list-style:none;padding:0;margin:0}}
    .trending-list li{{padding:8px 0;border-bottom:1px solid #f0f0f0;font-size:14px}}
    .site-footer{{background:#111;color:#ccc;padding:40px 0 20px;margin-top:48px}}
    .footer-inner{{max-width:1200px;margin:0 auto;padding:0 24px}}
    .footer-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:24px;margin-bottom:32px}}
    .footer-col h4{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#fff;margin:0 0 12px}}
    .footer-col ul{{list-style:none;padding:0;margin:0}}
    .footer-col li a{{font-size:13px;color:#aaa;text-decoration:none;display:block;padding:3px 0}}
    .footer-bottom{{border-top:1px solid #333;padding-top:20px;font-size:12px;color:#777}}
    .cookie-banner{{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:2px solid #e0e0e0;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;z-index:1000;box-shadow:0 -4px 12px rgba(0,0,0,.1)}}
    .cookie-banner p{{margin:0;font-size:14px;max-width:700px}}
    .cookie-actions{{display:flex;gap:12px;flex-shrink:0}}
    .btn{{padding:10px 20px;border-radius:4px;font-size:14px;font-weight:600;cursor:pointer;border:none}}
    .btn-primary{{background:#c00;color:#fff}}
    .btn-secondary{{background:#f0f0f0;color:#1a1a1a}}
    .newsletter-widget{{background:#f9f9f9;border:1px solid #e0e0e0;border-radius:4px;padding:20px;margin:24px 0}}
    .paywall-notice{{border:1px solid #c00;border-radius:4px;padding:20px;margin:24px 0;text-align:center}}
    .social-bar{{display:flex;gap:10px;margin:16px 0}}
    .social-btn{{padding:8px 16px;border-radius:4px;font-size:13px;font-weight:600;cursor:pointer;border:none;color:#fff}}
    .btn-fb{{background:#1877f2}}.btn-tw{{background:#000}}.btn-li{{background:#0a66c2}}.btn-wa{{background:#25d366;color:#000}}
    .ad-unit{{background:#f8f8f8;border:1px solid #e0e0e0;min-height:250px;display:flex;align-items:center;justify-content:center;margin:16px 0;font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px}}
    .breadcrumb{{font-size:13px;color:#666;margin-bottom:16px}}
    .article-meta{{display:flex;align-items:center;gap:16px;margin-bottom:16px;font-size:14px;color:#666}}
    .article-headline{{font-size:30px;font-weight:700;line-height:1.25;margin-bottom:12px}}
    .byline{{display:flex;align-items:center;gap:10px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #e0e0e0}}
  </style>
  <script>
    window.dataLayer=window.dataLayer||[];
    function gtag(){{dataLayer.push(arguments)}}
    gtag('js',new Date());
    gtag('config','G-XXXXXXXXXX');
    window.permutive=window.permutive||{{}};
    window.permutive.config={{projectId:'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',apiKey:'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',environment:'production'}};
    !function(e,o,n,i){{if(!e){{e=e||{{}};window.permutive=e;e.q=[];var t=function(){{return([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,function(e){{return(e^(window.crypto||window.msCrypto).getRandomValues(new Uint8Array(1))[0]&15>>e/4).toString(16)}})}};e.config=i||{{}};e.config.projectId=e.config.projectId||n;e.config.apiKey=e.config.apiKey||o;e.config.environment=e.config.environment||'production';Object.assign(e,{{addon:'web',track:function(o){{e.q.push({{type:'track',event:o,ts:Date.now()}});}},identify:function(o){{e.q.push({{type:'identify',identity:o,ts:Date.now()}});}}}});}}}};
    !function(f,b,e,v,n,t,s){{if(f.fbq)return;n=f.fbq=function(){{n.callMethod?n.callMethod.apply(n,arguments):n.queue.push(arguments)}};
    if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;
    t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}}(window,document,'script','https://connect.facebook.net/en_US/fbevents.js');
    fbq('init','123456789012345');fbq('track','PageView');
    var _comscore=_comscore||[];_comscore.push({{c1:'2',c2:'xxxxxxxx'}});
    (function(){{var s=document.createElement('script'),el=document.getElementsByTagName('script')[0];s.async=true;s.src='https://sb.scorecardresearch.com/beacon.js';el.parentNode.insertBefore(s,el)}})();
    (function(h,o,t,j,a,r){{h.hj=h.hj||function(){{(h.hj.q=h.hj.q||[]).push(arguments)}};h._hjSettings={{hjid:1234567,hjsv:6}};a=o.getElementsByTagName('head')[0];r=o.createElement('script');r.async=1;r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;a.appendChild(r)}})(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
    !function(){{var analytics=window.analytics=window.analytics||[];if(!analytics.initialize)if(analytics.invoked)window.console&&console.error&&console.error('Segment snippet included twice.');else{{analytics.invoked=!0;analytics.methods=['trackSubmit','trackClick','trackLink','trackForm','pageview','identify','reset','group','track','ready','alias','debug','page','once','off','on'];analytics.factory=function(e){{return function(){{var t=Array.prototype.slice.call(arguments);t.unshift(e);analytics.push(t);return analytics}}}};for(var t=0;t<analytics.methods.length;t++){{var e=analytics.methods[t];analytics[e]=analytics.factory(e)}}analytics.load=function(key,e){{var t=document.createElement('script');t.type='text/javascript';t.async=!0;t.src='https://cdn.segment.com/analytics.js/v1/'+key+'/analytics.min.js';var n=document.getElementsByTagName('script')[0];n.parentNode.insertBefore(t,n)}};analytics.load('YOUR_WRITE_KEY');analytics.page()}}}}();
  </script>
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
</head>
<body>
  <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-XXXXXXX" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
  <div class="cookie-banner" id="cookie-banner" role="dialog" aria-modal="true" aria-label="Cookie consent">
    <p>We and our 287 partners use cookies and similar technologies to store and access information on your device, develop and improve products, personalise ads and content, measure ad and content performance, and gain audience insights. You can choose to accept all cookies, accept only necessary cookies, or manage your preferences below.</p>
    <div class="cookie-actions">
      <button class="btn btn-primary" onclick="acceptAll()">Accept all</button>
      <button class="btn btn-secondary" onclick="acceptNecessary()">Necessary only</button>
      <button class="btn btn-secondary" onclick="managePrefs()">Manage preferences</button>
    </div>
  </div>
  <div class="breaking-bar">BREAKING: Follow live updates on today's top story &nbsp;&#8250;&nbsp; Subscribe now and save 50% for your first 3 months</div>
  <header class="site-header" role="banner">
    <div class="nav-wrapper">
      <a href="/" class="site-logo">The Daily Record</a>
      <nav role="navigation" aria-label="Main navigation">
        <ul class="nav-primary">
          <li><a href="/home">Home</a></li>
          <li><a href="/world">World</a></li>
          <li><a href="/us-news">US News</a></li>
          <li><a href="/politics">Politics</a></li>
          <li><a href="/business">Business</a></li>
          <li><a href="/technology">Technology</a></li>
          <li><a href="/science">Science</a></li>
          <li><a href="/health">Health</a></li>
          <li><a href="/entertainment">Entertainment</a></li>
          <li><a href="/sports">Sports</a></li>
          <li><a href="/travel">Travel</a></li>
          <li><a href="/food">Food</a></li>
          <li><a href="/lifestyle">Lifestyle</a></li>
          <li><a href="/opinion">Opinion</a></li>
          <li><a href="/investigations">Investigations</a></li>
          <li><a href="/video">Video</a></li>
          <li><a href="/podcasts">Podcasts</a></li>
          <li><a href="/newsletters">Newsletters</a></li>
        </ul>
      </nav>
      <div style="margin-left:auto;display:flex;gap:12px;align-items:center">
        <button class="btn btn-secondary" style="font-size:13px;padding:7px 14px">Sign in</button>
        <button class="btn btn-primary" style="font-size:13px;padding:7px 14px">Subscribe</button>
      </div>
    </div>
  </header>
  <div class="ad-unit" id="ad-leaderboard" data-ad-unit="/12345/record_article_top" data-ad-sizes="[[728,90],[970,90],[970,250]]" aria-label="Advertisement">Advertisement</div>
  <div class="layout">
    <main role="main" id="main-content">
      <nav aria-label="Breadcrumb" class="breadcrumb">
        <a href="/">Home</a> &rsaquo; <a href="/technology">Technology</a> &rsaquo; <span aria-current="page">{title[:60]}</span>
      </nav>
      <article itemscope itemtype="https://schema.org/NewsArticle">
        <div class="article-meta">
          <time datetime="2024-03-15T12:00:00Z" itemprop="datePublished">March 15, 2024</time>
          <span>5 min read</span>
          <a href="/technology">Technology</a>
        </div>
        <h1 class="article-headline" itemprop="headline">{title}</h1>
        <div class="byline" itemprop="author">
          <img src="/static/authors/staff.jpg" alt="" width="36" height="36" style="border-radius:50%"/>
          <span>By <a href="/authors/staff">Staff Reporter</a></span>
        </div>
        <div class="social-bar" aria-label="Share this article">
          <button class="social-btn btn-fb">Facebook</button>
          <button class="social-btn btn-tw">Twitter / X</button>
          <button class="social-btn btn-li">LinkedIn</button>
          <button class="social-btn btn-wa">WhatsApp</button>
          <button class="social-btn" style="background:#f0f0f0;color:#1a1a1a">Copy link</button>
        </div>
        <div class="article-body" itemprop="articleBody">
          {body}
        </div>
        <div class="paywall-notice">
          <p>You have read 3 of your 5 free articles this month.</p>
          <a href="/subscribe" class="btn btn-primary">Subscribe for unlimited access from $1/week</a>
          <a href="/login" class="btn btn-secondary" style="margin-top:8px">Already a subscriber? Sign in</a>
        </div>
        <div class="social-bar" aria-label="Share this article">
          <button class="social-btn btn-fb">Share on Facebook</button>
          <button class="social-btn btn-tw">Share on Twitter</button>
          <button class="social-btn btn-li">Share on LinkedIn</button>
        </div>
        <div class="article-tags" style="margin-top:16px">
          <span style="font-size:13px;font-weight:600;margin-right:8px">Topics:</span>
          <a href="/topics/technology" style="font-size:13px;margin-right:8px;color:#c00">Technology</a>
          <a href="/topics/business" style="font-size:13px;margin-right:8px;color:#c00">Business</a>
          <a href="/topics/economy" style="font-size:13px;margin-right:8px;color:#c00">Economy</a>
          <a href="/topics/ai" style="font-size:13px;margin-right:8px;color:#c00">Artificial Intelligence</a>
        </div>
      </article>
      <div class="newsletter-widget">
        <h3 style="margin:0 0 8px;font-size:18px">Get the Morning Briefing</h3>
        <p style="font-size:14px;color:#555;margin:0 0 12px">The five stories you need to read before 9am, direct to your inbox every weekday.</p>
        <form onsubmit="return false;" style="display:flex;gap:8px">
          <input type="email" placeholder="Enter your email" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:4px;font-size:14px"/>
          <button type="submit" class="btn btn-primary">Subscribe free</button>
        </form>
        <p style="font-size:12px;color:#888;margin:8px 0 0">By subscribing you agree to our <a href="/privacy">Privacy Policy</a> and <a href="/terms">Terms of Service</a>.</p>
      </div>
    </main>
    <aside role="complementary" aria-label="Sidebar">
      <div class="sidebar-widget">
        <h3>Trending Now</h3>
        <ol class="trending-list">
          <li><a href="/article/markets-react-to-fed-decision">Markets react to Fed's surprise rate decision — what it means for you</a></li>
          <li><a href="/article/tech-layoffs-second-wave">Second wave of tech layoffs hits major Silicon Valley firms</a></li>
          <li><a href="/article/climate-bill-passes-senate">Historic climate bill passes Senate with bipartisan support</a></li>
          <li><a href="/article/new-ai-model-benchmark">New AI model shatters records across every major benchmark</a></li>
          <li><a href="/article/housing-market-update">Housing affordability reaches all-time low in 14 major cities</a></li>
          <li><a href="/article/drug-pricing-reform">Congress passes sweeping drug pricing reform legislation</a></li>
          <li><a href="/article/election-latest-polls">Election 2024: latest polling in seven key battleground states</a></li>
          <li><a href="/article/inflation-data-release">Inflation data surprise: CPI comes in lower than forecast</a></li>
        </ol>
      </div>
      <div class="ad-unit" id="ad-sidebar" data-ad-unit="/12345/record_article_sidebar" data-ad-sizes="[[300,250],[300,600]]" aria-label="Advertisement">Advertisement</div>
      <div class="sidebar-widget">
        <h3>Most Read This Week</h3>
        <ol class="trending-list">
          <li><a href="/article/1">Central bank signals end of rate hike cycle; markets surge on the news</a></li>
          <li><a href="/article/2">Major retailer to close 400 stores as online sales continue to dominate</a></li>
          <li><a href="/article/3">Scientists announce breakthrough in early Alzheimer's detection</a></li>
          <li><a href="/article/4">National security agency confirms breach of government email servers</a></li>
          <li><a href="/article/5">College admissions hit record low acceptance rates amid applications surge</a></li>
          <li><a href="/article/6">Supreme Court to hear landmark case on social media regulation</a></li>
        </ol>
      </div>
      <div class="sidebar-widget">
        <h3>From Our Partners</h3>
        <ul class="trending-list">
          <li><a href="/sponsored/1">How leading companies are cutting infrastructure costs by 40% with cloud</a></li>
          <li><a href="/sponsored/2">The investment strategy that outperformed the S&amp;P 500 three years running</a></li>
          <li><a href="/sponsored/3">Why cybersecurity leaders are switching to zero-trust architecture now</a></li>
        </ul>
      </div>
    </aside>
  </div>
  <section aria-label="Recommended articles" style="background:#f9f9f9;padding:40px 0;border-top:1px solid #e0e0e0">
    <div style="max-width:1200px;margin:0 auto;padding:0 24px">
      <h2 style="font-size:20px;font-weight:700;margin:0 0 24px">More from Technology</h2>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">
        <div><img src="/img/rec/1.jpg" alt="Article thumbnail" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px" loading="lazy" width="280" height="157"/><span style="font-size:12px;color:#c00;font-weight:600;display:block;margin:8px 0 4px">TECHNOLOGY</span><a href="/rec/1" style="font-size:15px;font-weight:600;line-height:1.3;text-decoration:none;color:#1a1a1a">Chip shortage eases but supply chain experts warn recovery is fragile</a><time style="font-size:12px;color:#888;display:block;margin-top:6px">2 hours ago</time></div>
        <div><img src="/img/rec/2.jpg" alt="Article thumbnail" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px" loading="lazy" width="280" height="157"/><span style="font-size:12px;color:#c00;font-weight:600;display:block;margin:8px 0 4px">BUSINESS</span><a href="/rec/2" style="font-size:15px;font-weight:600;line-height:1.3;text-decoration:none;color:#1a1a1a">Big Tech antitrust investigation enters new phase with Senate subpoenas</a><time style="font-size:12px;color:#888;display:block;margin-top:6px">4 hours ago</time></div>
        <div><img src="/img/rec/3.jpg" alt="Article thumbnail" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px" loading="lazy" width="280" height="157"/><span style="font-size:12px;color:#c00;font-weight:600;display:block;margin:8px 0 4px">BUSINESS</span><a href="/rec/3" style="font-size:15px;font-weight:600;line-height:1.3;text-decoration:none;color:#1a1a1a">Electric vehicle adoption stalls in rural markets as charging gaps persist</a><time style="font-size:12px;color:#888;display:block;margin-top:6px">5 hours ago</time></div>
        <div><img src="/img/rec/4.jpg" alt="Article thumbnail" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:4px" loading="lazy" width="280" height="157"/><span style="font-size:12px;color:#c00;font-weight:600;display:block;margin:8px 0 4px">POLITICS</span><a href="/rec/4" style="font-size:15px;font-weight:600;line-height:1.3;text-decoration:none;color:#1a1a1a">Bipartisan infrastructure bill clears final procedural hurdle in Senate</a><time style="font-size:12px;color:#888;display:block;margin-top:6px">7 hours ago</time></div>
      </div>
    </div>
  </section>
  <footer class="site-footer" role="contentinfo">
    <div class="footer-inner">
      <div class="footer-grid">
        <nav aria-label="Footer sections">
          <h4>Sections</h4>
          <ul>
            <li><a href="/world">World</a></li>
            <li><a href="/us-news">US News</a></li>
            <li><a href="/politics">Politics</a></li>
            <li><a href="/business">Business</a></li>
            <li><a href="/technology">Technology</a></li>
            <li><a href="/science">Science</a></li>
            <li><a href="/health">Health</a></li>
            <li><a href="/entertainment">Entertainment</a></li>
            <li><a href="/sports">Sports</a></li>
            <li><a href="/opinion">Opinion</a></li>
            <li><a href="/investigations">Investigations</a></li>
          </ul>
        </nav>
        <nav aria-label="Footer services">
          <h4>Services</h4>
          <ul>
            <li><a href="/newsletters">Newsletters</a></li>
            <li><a href="/podcasts">Podcasts</a></li>
            <li><a href="/video">Video</a></li>
            <li><a href="/app">Mobile app</a></li>
            <li><a href="/rss">RSS feeds</a></li>
            <li><a href="/alerts">News alerts</a></li>
            <li><a href="/archive">Archive</a></li>
            <li><a href="/crossword">Crossword</a></li>
          </ul>
        </nav>
        <nav aria-label="Footer company">
          <h4>Company</h4>
          <ul>
            <li><a href="/about">About us</a></li>
            <li><a href="/careers">Careers</a></li>
            <li><a href="/press">Press room</a></li>
            <li><a href="/advertise">Advertise with us</a></li>
            <li><a href="/contact">Contact us</a></li>
            <li><a href="/masthead">Masthead</a></li>
            <li><a href="/ethics">Ethics policy</a></li>
            <li><a href="/corrections">Corrections</a></li>
          </ul>
        </nav>
        <nav aria-label="Footer support">
          <h4>Support</h4>
          <ul>
            <li><a href="/help">Help centre</a></li>
            <li><a href="/subscribe">Subscribe</a></li>
            <li><a href="/manage">Manage subscription</a></li>
            <li><a href="/gift">Gift subscription</a></li>
            <li><a href="/student">Student discount</a></li>
            <li><a href="/accessibility">Accessibility</a></li>
          </ul>
        </nav>
        <nav aria-label="Footer legal">
          <h4>Legal</h4>
          <ul>
            <li><a href="/privacy">Privacy policy</a></li>
            <li><a href="/terms">Terms of service</a></li>
            <li><a href="/cookies">Cookie policy</a></li>
            <li><a href="/do-not-sell">Do not sell my info</a></li>
            <li><a href="/ccpa">CCPA rights</a></li>
            <li><a href="/gdpr">GDPR</a></li>
            <li><a href="/dmca">DMCA</a></li>
          </ul>
        </nav>
      </div>
      <div class="footer-bottom">
        <p>&copy; 2024 Daily Record Media Group. All rights reserved. Reproduction of material from any Daily Record pages without written permission is strictly prohibited. &quot;Daily Record&quot; and the Daily Record logo are registered trademarks.</p>
        <p style="margin-top:8px"><a href="/privacy" style="color:#aaa">Privacy</a> &middot; <a href="/terms" style="color:#aaa">Terms</a> &middot; <a href="/cookies" style="color:#aaa">Cookies</a> &middot; <a href="/accessibility" style="color:#aaa">Accessibility</a></p>
      </div>
    </div>
  </footer>
  <script type="application/ld+json">
    {{"@context":"https://schema.org","@type":"NewsArticle","headline":"{title}","datePublished":"2024-03-15T12:00:00Z","author":{{"@type":"Person","name":"Staff Reporter"}},"publisher":{{"@type":"Organization","name":"The Daily Record","logo":{{"@type":"ImageObject","url":"https://dailyrecord.com/logo.png"}}}}}}
  </script>
</body>
</html>"""


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────


def load_pages() -> list[str]:
    print(f"\nLoading dataset (cc_news): {N_PAGES:,} pages")

    ds = load_dataset("cc_news", split=f"train[:{N_PAGES}]")

    pages = []
    for row in ds:
        title = row.get("title") or "Untitled"
        body = row.get("maintext") or row.get("text") or ""

        paragraphs = body.split("\n\n")
        body_html = "\n".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())

        pages.append(wrap_html(title[:200], body_html))

    total_mb = sum(len(x) for x in pages) / 1e6
    print(f"✓ Loaded {len(pages):,} pages")
    print(f"✓ Total HTML size: {total_mb:.1f} MB\n")
    return pages


# ─────────────────────────────────────────────
# Python extraction baseline
# ─────────────────────────────────────────────

NOISE = {"script", "style", "nav", "header", "footer", "aside", "noscript", "form"}


def py_extract(html: str) -> str:
    tree = HTMLParser(html)
    for tag in NOISE:
        for node in tree.css(tag):
            node.decompose()
    return " ".join(tree.text(separator=" ").split())


def py_batch(htmls: list[str]) -> list[str]:
    return [py_extract(x) for x in htmls]


# ─────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ─────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────


def bench(fn, *args) -> float:
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return median(times)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main() -> None:
    html_pages = load_pages()

    # ── extraction ───────────────────────────────────────────────
    print("Extracting clean text (Rust, parallel)...")
    rust_clean = batch_extract_clean_text(html_pages)

    # ── token stats ──────────────────────────────────────────────
    print("Counting tokens (tiktoken cl100k_base)...")

    html_tokens = []
    clean_tokens = []

    # tiktoken on 100k pages takes a few minutes — progress every 10k
    for i, (html, clean) in enumerate(zip(html_pages, rust_clean, strict=False)):
        html_tokens.append(count_tokens(html))
        clean_tokens.append(count_tokens(clean))
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1:,} / {N_PAGES:,} tokenized...")

    avg_html = mean(html_tokens)
    avg_clean = mean(clean_tokens)
    reduction = (1 - avg_clean / avg_html) * 100

    print("\nToken statistics (100k pages)")
    print("─" * 44)
    print(f"Avg HTML tokens per page:   {avg_html:,.0f}")
    print(f"Avg clean tokens per page:  {avg_clean:,.0f}")
    print(f"Token reduction:            {reduction:.1f}%")
    print(f"Noise tokens per page:      {avg_html - avg_clean:,.0f}")
    print(f"Signal ratio:               {avg_clean / avg_html * 100:.1f}%")

    # percentile breakdown
    sorted_reductions = sorted(
        (1 - c / h) * 100 for h, c in zip(html_tokens, clean_tokens, strict=False) if h > 0
    )
    n = len(sorted_reductions)
    print("\nReduction distribution:")
    print(f"  p10: {sorted_reductions[n // 10]:.1f}%")
    print(f"  p50: {sorted_reductions[n // 2]:.1f}%")
    print(f"  p90: {sorted_reductions[n * 9 // 10]:.1f}%")
    print(f"  stdev: {stdev(sorted_reductions):.1f}%")

    # ── cost estimation ──────────────────────────────────────────
    monthly_pages = 100_000
    html_total = avg_html * monthly_pages
    clean_total = avg_clean * monthly_pages
    cached_total = clean_total * 0.20  # 80% cache hit rate

    print(f"\nLLM cost at {monthly_pages:,} pages/month")
    print("─" * 52)
    header = f"{'Format':<32} {'Tokens':>14}"
    for model in LLM_PRICES:
        header += f"  {model + '/mo':>14}"
    print(header)
    print("─" * 52)

    for label, tokens in [
        ("Raw HTML", html_total),
        ("Clean text", clean_total),
        ("Clean + cache (80% hit rate)", cached_total),
    ]:
        row = f"{label:<32} {tokens:>14,.0f}"
        for price in LLM_PRICES.values():
            cost = tokens / 1_000_000 * price
            row += f"  ${cost:>13,.2f}"
        print(row)

    raw_gpt4o = html_total / 1_000_000 * list(LLM_PRICES.values())[0]
    cache_gpt4o = cached_total / 1_000_000 * list(LLM_PRICES.values())[0]
    print(f"\nMonthly saving (raw HTML → clean + cache): ${raw_gpt4o - cache_gpt4o:,.2f}")
    print(f"Annual saving:                             ${(raw_gpt4o - cache_gpt4o) * 12:,.2f}")

    # ── speed comparison ─────────────────────────────────────────
    SAMPLE = 10_000
    PY_SAMPLE = 1_000  # BeautifulSoup/selectolax scaled from 1k → 10k for display
    sample = html_pages[:SAMPLE]
    py_sample = html_pages[:PY_SAMPLE]

    print(f"\nExtraction speed ({RUNS} runs, median)")
    print("─" * 64)
    print(f"{'Operation':<28} {'Python':>14} {'Rust':>14} {'Speedup':>8}")
    print("─" * 64)

    # warmup
    batch_extract_clean_text(sample)
    py_batch(py_sample)

    py_t = bench(py_batch, py_sample)
    py_t_scaled = py_t * (SAMPLE / PY_SAMPLE)  # scale to 10k for display
    rs_t = bench(batch_extract_clean_text, sample)

    speedup = py_t_scaled / rs_t
    py_rate = int(PY_SAMPLE / (py_t / 1000))
    rs_rate = int(SAMPLE / (rs_t / 1000))

    print(
        f"{'HTML extraction (10k pages)':<28}"
        f"{py_t_scaled:>13,.1f}ms"
        f"{rs_t:>13,.1f}ms"
        f"{speedup:>7.1f}x"
    )
    print("─" * 64)
    print(f"Python: {py_rate:,} pages/s (serial, selectolax)")
    print(f"Rust:   {rs_rate:,} pages/s (parallel, Rayon)")

    print("\nMethodology: real cc_news articles wrapped in realistic")
    print("news-site HTML (tracking scripts, full nav, sidebar, footer).")
    print("Token counting: tiktoken cl100k_base (exact, not estimated).")
    print("Source: github.com/yourusername/crawlkit\n")


if __name__ == "__main__":
    from statistics import stdev

    main()
