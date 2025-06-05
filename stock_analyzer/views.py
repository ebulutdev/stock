from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
import yfinance as yf
from django.conf import settings
import time
from django.core.cache.backends.base import InvalidCacheBackendError
from requests.exceptions import RequestException
from curl_cffi import requests as curl_requests
import logging
import pandas as pd
import json
from random import uniform
from time import sleep
from datetime import datetime, timedelta
import google.generativeai as genai
import numpy as np

class RateLimitHandler:
    def __init__(self):
        self.last_request_time = {}
        self.min_delay = 1.0  # Minimum bekleme süresi
        self.max_delay = 2.0  # Maximum bekleme süresi
        self.request_counts = {}
        self.window_size = 60  # 60 saniyelik pencere
        self.max_requests = 50  # Pencere başına maksimum istek
        
    def can_make_request(self, symbol):
        current_time = time.time()
        # Eski istekleri temizle
        self.cleanup_old_requests(current_time)
        
        # Symbol için istek sayısını kontrol et
        requests_in_window = len(self.request_counts.get(symbol, []))
        return requests_in_window < self.max_requests
    
    def cleanup_old_requests(self, current_time):
        for symbol in list(self.request_counts.keys()):
            self.request_counts[symbol] = [
                timestamp for timestamp in self.request_counts.get(symbol, [])
                if current_time - timestamp < self.window_size
            ]
    
    def record_request(self, symbol):
        current_time = time.time()
        if symbol not in self.request_counts:
            self.request_counts[symbol] = []
        self.request_counts[symbol].append(current_time)
    
    def wait_if_needed(self, symbol):
        current_time = time.time()
        
        # Rate limit kontrolü
        if not self.can_make_request(symbol):
            sleep_time = self.window_size - (current_time - min(self.request_counts[symbol]))
            time.sleep(max(sleep_time, self.min_delay))
        
        # Normal bekleme süresi
        if symbol in self.last_request_time:
            elapsed = current_time - self.last_request_time[symbol]
            if elapsed < self.min_delay:
                sleep_time = uniform(self.min_delay, self.max_delay)
                time.sleep(sleep_time)
        
        self.last_request_time[symbol] = current_time
        self.record_request(symbol)

rate_limit_handler = RateLimitHandler()

def normalize_symbol(symbol):
    symbol = symbol.upper().strip()
    if symbol.endswith('.IS'):
        symbol = symbol[:-3]
    return symbol

def analyze_price_history(history_df, current_price, tolerance=0.02):
    """
    Geçmiş fiyat seviyelerinde hissenin davranışını analiz eder
    tolerance: fiyat eşleşmesi için tolerans yüzdesi (örn: 0.02 = %2)
    """
    cache_key = f"price_history_{hash(str(list(history_df.index)))}_{current_price}_{tolerance}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    if history_df.empty:
        return None

    # Fiyat aralığını belirle
    price_min = current_price * (1 - tolerance)
    price_max = current_price * (1 + tolerance)
    
    # Benzer fiyat seviyelerini bul
    similar_prices = history_df[
        (history_df['Close'] >= price_min) & 
        (history_df['Close'] <= price_max)
    ].copy()
    
    if similar_prices.empty:
        return None

    results = []
    for date in similar_prices.index:
        if date >= history_df.index[-1]:  # Son günü atlayalım
            continue
        # Sonraki günlerdeki değişimi hesapla (1, 5, 22 gün)
        next_days = {}
        for days in [1, 5, 22]:
            future_idx = history_df.index.get_loc(date) + days
            if future_idx < len(history_df):
                future_price = history_df['Close'].iloc[future_idx]
                change_pct = ((future_price - similar_prices.loc[date, 'Close']) / similar_prices.loc[date, 'Close']) * 100
                next_days[days] = {
                    'price': future_price,
                    'change_pct': change_pct
                }
        results.append({
            'date': date.strftime('%d.%m.%Y'),
            'price': similar_prices.loc[date, 'Close'],
            'next_days': next_days
        })
    cache.set(cache_key, results, 3600)
    return results

def cache_stock_data(symbol, data, cache_duration=7200):  # 2 saat
    """Hisse verilerini önbelleğe al"""
    cache_key = f'stock_data_{symbol}'
    cache.set(cache_key, {
        'data': data,
        'timestamp': time.time()
    }, cache_duration)

def get_cached_stock_data(symbol):
    """Önbellekteki hisse verilerini getir"""
    cache_key = f'stock_data_{symbol}'
    cached = cache.get(cache_key)
    if cached:
        # 1 saat içindeyse önbellekten getir
        if time.time() - cached['timestamp'] < 3600:
            return cached['data']
    return None

def get_stock_data(symbol):
    """Geliştirilmiş hisse veri çekme fonksiyonu"""
    try:
        # Önbellekten veri kontrolü
        cached_data = get_cached_stock_data(symbol)
        if cached_data:
            return cached_data
    except (InvalidCacheBackendError, Exception) as e:
        logging.warning(f"Cache hatası: {e}")

    rate_limit_handler.wait_if_needed(symbol)

    try:
        norm_symbol = normalize_symbol(symbol)
        session = curl_requests.Session(impersonate="chrome110")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                stock = yf.Ticker(f"{norm_symbol}.IS", session=session)
                price = get_price_with_fallback(stock)
                
                if price and price > 0:
                    try:
                        # Son 1 yıllık veriyi çek
                        history = stock.history(period="1y")
                        
                        if not history.empty:
                            # Fiyat analizi yap
                            price_analysis = analyze_price_history(history, price)
                            
                            # Teknik göstergeleri hesapla
                            # Hareketli Ortalamalar
                            history['MA50'] = history['Close'].rolling(window=50).mean()
                            history['MA200'] = history['Close'].rolling(window=200).mean()
                            
                            # RSI
                            delta = history['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            history['RSI'] = 100 - (100 / (1 + rs))
                            
                            # MACD
                            exp1 = history['Close'].ewm(span=12, adjust=False).mean()
                            exp2 = history['Close'].ewm(span=26, adjust=False).mean()
                            history['MACD'] = exp1 - exp2
                            history['Signal'] = history['MACD'].ewm(span=9, adjust=False).mean()
                            
                            # Bollinger Bands
                            history['BB_middle'] = history['Close'].rolling(window=20).mean()
                            std = history['Close'].rolling(window=20).std()
                            history['BB_upper'] = history['BB_middle'] + (std * 2)
                            history['BB_lower'] = history['BB_middle'] - (std * 2)
                            
                            # NaN değerleri None'a çevir
                            history = history.where(pd.notnull(history), None)
                            
                            # Timestamp'leri string'e çevir
                            def convert_timestamps(obj):
                                if isinstance(obj, pd.Timestamp):
                                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                                elif isinstance(obj, dict):
                                    return {str(k): convert_timestamps(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_timestamps(item) for item in obj]
                                return obj

                            # Özet istatistikler
                            summary = {
                                'total_occurrences': len(price_analysis) if price_analysis else 0,
                                'average_next_day': 0,
                                'average_next_week': 0,
                                'average_next_month': 0,
                                'positive_next_day': 0,
                                'positive_next_week': 0,
                                'positive_next_month': 0
                            }
                            
                            if price_analysis:
                                for analysis in price_analysis:
                                    next_days = analysis['next_days']
                                    if 1 in next_days:
                                        summary['average_next_day'] += next_days[1]['change_pct']
                                        if next_days[1]['change_pct'] > 0:
                                            summary['positive_next_day'] += 1
                                    if 5 in next_days:
                                        summary['average_next_week'] += next_days[5]['change_pct']
                                        if next_days[5]['change_pct'] > 0:
                                            summary['positive_next_week'] += 1
                                    if 22 in next_days:
                                        summary['average_next_month'] += next_days[22]['change_pct']
                                        if next_days[22]['change_pct'] > 0:
                                            summary['positive_next_month'] += 1
                                
                                total = len(price_analysis)
                                if total > 0:
                                    summary['average_next_day'] /= total
                                    summary['average_next_week'] /= total
                                    summary['average_next_month'] /= total
                            
                            # Finansal metrikleri ekle
                            try:
                                info = stock.info
                                financials = {
                                    'Revenue': info.get('totalRevenue'),
                                    'Profit Margin': info.get('profitMargins'),
                                    'Operating Margin': info.get('operatingMargins'),
                                    'Current Ratio': info.get('currentRatio'),
                                    'Debt to Equity': info.get('debtToEquity'),
                                    'Return on Equity': info.get('returnOnEquity'),
                                    'Price to Earnings': info.get('trailingPE'),
                                    'Price to Book': info.get('priceToBook')
                                }
                            except Exception as e:
                                logging.warning(f"Finansal metrik hatası: {e}")
                                financials = {}
                            
                            data = {
                                'current_price': price,
                                'timestamp': time.time(),
                                'price_history': price_analysis,
                                'summary': summary,
                                'technical_indicators': {
                                    'ma50': convert_timestamps(history['MA50'].to_dict()),
                                    'ma200': convert_timestamps(history['MA200'].to_dict()),
                                    'rsi': convert_timestamps(history['RSI'].to_dict()),
                                    'macd': {
                                        'macd': convert_timestamps(history['MACD'].to_dict()),
                                        'signal': convert_timestamps(history['Signal'].to_dict())
                                    },
                                    'bollinger_bands': {
                                        'upper': convert_timestamps(history['BB_upper'].to_dict()),
                                        'middle': convert_timestamps(history['BB_middle'].to_dict()),
                                        'lower': convert_timestamps(history['BB_lower'].to_dict())
                                    }
                                },
                                'financials': financials,
                                'volume_data': convert_timestamps(history['Volume'].to_dict())
                            }
                            
                            # Gemini analizi ekle
                            if price_analysis:
                                try:
                                    gemini_comment = gemini_flash_comment(summary, data['technical_indicators'], price_analysis)
                                    data['gemini_analysis'] = parse_gemini_response(gemini_comment)
                                except Exception as e:
                                    logging.warning(f"Gemini analiz hatası: {e}")
                            
                            # Veriyi önbelleğe al
                            try:
                                cache_stock_data(symbol, data)
                            except Exception as e:
                                logging.warning(f"Cache yazma hatası: {e}")
                            return data
                    except Exception as e:
                        logging.warning(f"Geçmiş veri analizi hatası: {e}")
                        # Sadece güncel fiyatı döndür
                        return {
                            'current_price': price,
                            'timestamp': time.time()
                        }
                
            except Exception as e:
                if is_rate_limit_error(e):
                    retry_count += 1
                    if retry_count < max_retries:
                        sleep_time = exponential_backoff(retry_count)
                        time.sleep(sleep_time)
                        session = curl_requests.Session(impersonate="chrome110")
                        continue
                    return {'error': 'Çok sık sorgu yaptınız, lütfen birkaç dakika bekleyip tekrar deneyiniz.'}
                raise
            
            retry_count += 1
            
        return {'error': f'{norm_symbol}.IS için fiyat verisi alınamadı.'}
        
    except Exception as e:
        logging.error(f"Beklenmeyen hata: {e}, symbol: {symbol}")
        return {'error': f'Beklenmeyen bir hata oluştu ({type(e).__name__}).'}

def get_price_with_fallback(stock):
    """Farklı metodlarla fiyat almayı dener"""
    try:
        return stock.fast_info['last_price']
    except Exception:
        try:
            return stock.info.get('currentPrice')
        except Exception:
            try:
                history = stock.history(period='1d')
                if not history.empty:
                    return history['Close'].iloc[-1]
            except Exception:
                return None
    return None

def exponential_backoff(retry_count):
    """Üstel geri çekilme süresi hesaplar"""
    return min(300, (2 ** retry_count) + uniform(0, 1))

def is_rate_limit_error(e):
    """Rate limit hatası olup olmadığını kontrol eder"""
    if hasattr(e, 'response'):
        return getattr(e.response, 'status_code', None) == 429
    return '429' in str(e) or 'rate limit' in str(e).lower()

def fetch_yahoo_history(symbol, start, end, interval="1d"):
    """Geliştirilmiş tarihsel veri çekme fonksiyonu"""
    cache_key = f'history_{symbol}_{start}_{end}_{interval}'
    
    try:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
    except Exception as e:
        logging.warning(f"Cache hatası: {e}")

    rate_limit_handler.wait_if_needed(symbol)
    
    norm_symbol = normalize_symbol(symbol)
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{norm_symbol}.IS"
    
    params = {
        'period1': start,
        'period2': end,
        'interval': interval,
        'events': 'history'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/110.0.0.0 Safari/537.36'
    }
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            session = curl_requests.Session(impersonate="chrome110")
            resp = session.get(url, params=params, headers=headers)
            
            if resp.status_code == 200:
                data = resp.json()
                df = process_yahoo_response(data)
                
                try:
                    cache.set(cache_key, df.to_dict(), settings.CACHE_TTL)
                except Exception as e:
                    logging.warning(f"Cache yazma hatası: {e}")
                
                return df
                
            elif resp.status_code == 429:
                retry_count += 1
                if retry_count < max_retries:
                    sleep_time = exponential_backoff(retry_count)
                    time.sleep(sleep_time)
                    continue
                return "Çok sık sorgu yaptınız, lütfen birkaç dakika bekleyip tekrar deneyiniz."
                
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                sleep_time = exponential_backoff(retry_count)
                time.sleep(sleep_time)
                continue
            return f"Hata: {str(e)}"
    
    return "Veri çekilemedi, lütfen daha sonra tekrar deneyiniz."

def process_yahoo_response(data):
    """Yahoo Finance yanıtını işler"""
    timestamps = data['chart']['result'][0]['timestamp']
    quotes = data['chart']['result'][0]['indicators']['quote'][0]
    
    df = pd.DataFrame({
        'Date': [datetime.fromtimestamp(ts) for ts in timestamps],
        'Close': quotes['close'],
        'Open': quotes.get('open', [None] * len(timestamps)),
        'High': quotes.get('high', [None] * len(timestamps)),
        'Low': quotes.get('low', [None] * len(timestamps)),
        'Volume': quotes.get('volume', [None] * len(timestamps))
    })
    
    df.set_index('Date', inplace=True)
    return df

USER_RATE_LIMIT_KEY = 'user_rate_limit_{ip}'
USER_RATE_LIMIT_SECONDS = 60  # 1 dakika

def gemini_flash_comment(summary, technical_indicators, price_history):
    """
    Gemini ile detaylı analiz yapan fonksiyon
    """
    cache_key = f"gemini_analysis_{hash(json.dumps(summary, sort_keys=True))}_{hash(json.dumps(technical_indicators, sort_keys=True))}_{hash(json.dumps(price_history, sort_keys=True))}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash") # Pro modeli kullanalım
        # Teknik göstergeleri analiz et
        latest_tech = get_latest_technical_data(technical_indicators)
        # Özet istatistikleri formatla
        stats = format_summary_stats(summary)
        # Fiyat geçmişini formatla
        price_data = format_price_history(price_history)
        prompt = f'''
        -Senaryo analizi yapın.
        -Fiyat seviyesi analizi yapın.
        -Teknik göstergeleri analiz edin.
        -Geçmiş fiyat hareketlerini ve seviyelerini analiz edin.
        -Tüm geçmiş verilerden toplu bir sonuç ve genel eğilim çıkarın, özetleyin.
        -Analizin başında veya sonunda <div style="background:#e8f5e9;border-left:5px solid #388e3c;padding:10px;margin-bottom:10px;"><b>Genel Sonuç / Toplu Eğilim:</b> ...</div> şeklinde, geçmiş verilerin ve teknik göstergelerin ortak değerlendirmesini özetleyen bir kutu oluşturun.
        -yatırım tavsiyesi vermeyin ve bunu açıkca belirtin.
        -renkli yazılar kullanın.
        -önemli rakamları vurgulayın.
        -önemli sinyalleri vurgulayın.
        -önemli seviyeleri vurgulayın.
        
<analysis_request>
    <summary_stats>
        {stats}
    </summary_stats>
    <technical_data>
        {latest_tech}
    </technical_data>
    <price_history>
        {price_data}
    </price_history>
</analysis_request>

Lütfen yukarıdaki verileri kullanarak aşağıdaki HTML formatında RENKLİ ve TABLOLU bir analiz yapın:

<analysis>
    <general_overview>
        <div style="background:#e3f2fd;padding:10px;border-radius:8px;">
            <b>Genel Durum:</b>
            <ul>
                <li>Toplam benzer seviye: <span style="color:#1976d2;">[rakam]</span></li>
                <li>Ortalama getiriler: <span style="color:#388e3c;">[pozitif]</span>, <span style="color:#d32f2f;">[negatif]</span></li>
            </ul>
        </div>
    </general_overview>

    <technical_analysis>
        <table style="width:100%%;border-collapse:collapse;">
            <tr style="background:#1976d2;color:white;">
                <th>Gösterge</th><th>Değer</th><th>Yorum</th>
            </tr>
            <tr>
                <td>MA50</td>
                <td style="color:#388e3c;">[MA50]</td>
                <td>[MA50 yorumu]</td>
            </tr>
            <tr>
                <td>MA200</td>
                <td style="color:#388e3c;">[MA200]</td>
                <td>[MA200 yorumu]</td>
            </tr>
            <tr>
                <td>RSI</td>
                <td style="color:#d32f2f;">[RSI]</td>
                <td>[RSI yorumu]</td>
            </tr>
            <tr>
                <td>MACD</td>
                <td style="color:#1976d2;">[MACD]</td>
                <td>[MACD yorumu]</td>
            </tr>
        </table>
    </technical_analysis>

    <scenarios>
        <div style="display:flex;gap:10px;">
            <div style="background:#c8e6c9;padding:10px;border-radius:8px;flex:1;">
                <b>Yüksek Olasılıklı Senaryo</b><br>
                <span style="color:#388e3c;">[Senaryo metni]</span>
            </div>
            <div style="background:#fff9c4;padding:10px;border-radius:8px;flex:1;">
                <b>Orta Olasılıklı Senaryo</b><br>
                <span style="color:#fbc02d;">[Senaryo metni]</span>
            </div>
            <div style="background:#ffcdd2;padding:10px;border-radius:8px;flex:1;">
                <b>Düşük Olasılıklı Senaryo</b><br>
                <span style="color:#d32f2f;">[Senaryo metni]</span>
            </div>
        </div>
    </scenarios>

    <key_levels>
        <table style="width:100%%;border-collapse:collapse;">
            <tr style="background:#1976d2;color:white;">
                <th>Seviye</th><th>Tip</th>
            </tr>
            <tr>
                <td style="color:#388e3c;">[Destek]</td>
                <td>Destek</td>
            </tr>
            <tr>
                <td style="color:#d32f2f;">[Direnç]</td>
                <td>Direnç</td>
            </tr>
        </table>
    </key_levels>
</analysis>

Not: Tüm önemli rakamları ve sinyalleri uygun renklerle vurgulayın. Tablo ve kutu kullanın. Sadece istatistiksel ve teknik analiz yapın, yatırım tavsiyesi vermeyin.
'''
        response = model.generate_content(
            prompt,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                }
            ],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        if not response.text:
            raise ValueError("Boş yanıt alındı")
        cache.set(cache_key, response.text.strip(), 3600)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini analiz hatası: {str(e)}", exc_info=True)
        return generate_fallback_analysis(summary, price_history)

def format_value(val):
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"

def get_latest_technical_data(technical_indicators):
    """En son teknik gösterge verilerini formatlar"""
    try:
        ma50_values = list(technical_indicators['ma50'].values())
        ma200_values = list(technical_indicators['ma200'].values())
        rsi_values = list(technical_indicators['rsi'].values())
        macd_values = list(technical_indicators['macd']['macd'].values())
        signal_values = list(technical_indicators['macd']['signal'].values())
        
        latest_data = {
            'MA50': ma50_values[-1] if ma50_values else None,
            'MA200': ma200_values[-1] if ma200_values else None,
            'RSI': rsi_values[-1] if rsi_values else None,
            'MACD': macd_values[-1] if macd_values else None,
            'Signal': signal_values[-1] if signal_values else None
        }
        
        return f"""
            MA50: {format_value(latest_data['MA50'])}
            MA200: {format_value(latest_data['MA200'])}
            RSI: {format_value(latest_data['RSI'])}
            MACD: {format_value(latest_data['MACD'])}
            Signal: {format_value(latest_data['Signal'])}
        """
    except Exception as e:
        logging.error(f"Teknik veri formatlama hatası: {str(e)}")
        return "Teknik veriler formatlanamadı"

def format_summary_stats(summary):
    """Özet istatistikleri formatlar"""
    try:
        return f"""
            Toplam Benzer Seviye: {summary['total_occurrences']}
            Ortalama Getiriler:
            - Günlük: {summary['average_next_day']:.2f}%
            - Haftalık: {summary['average_next_week']:.2f}%
            - Aylık: {summary['average_next_month']:.2f}%
            
            Yükseliş Oranları:
            - Günlük: {(summary['positive_next_day'] / summary['total_occurrences'] * 100):.1f}%
            - Haftalık: {(summary['positive_next_week'] / summary['total_occurrences'] * 100):.1f}%
            - Aylık: {(summary['positive_next_month'] / summary['total_occurrences'] * 100):.1f}%
        """
    except Exception as e:
        logging.error(f"Özet formatlama hatası: {str(e)}")
        return "Özet istatistikler formatlanamadı"

def format_price_history(price_history):
    """Fiyat geçmişini formatlar"""
    try:
        if not price_history:
            return "Fiyat geçmişi bulunamadı"
        
        history_text = []
        for event in price_history[:5]:  # Son 5 olay
            history_text.append(f"""
                Tarih: {event['date']}
                Fiyat: {event['price']:.2f}
                Değişimler:
                - 1 Gün: {event['next_days'].get(1, {}).get('change_pct', 0):.2f}%
                - 1 Hafta: {event['next_days'].get(5, {}).get('change_pct', 0):.2f}%
                - 1 Ay: {event['next_days'].get(22, {}).get('change_pct', 0):.2f}%
            """)
        return "\n".join(history_text)
    except Exception as e:
        logging.error(f"Fiyat geçmişi formatlama hatası: {str(e)}")
        return "Fiyat geçmişi formatlanamadı"

def parse_gemini_response(response):
    """XML/HTML formatındaki Gemini yanıtını parse eder"""
    try:
        # Temel bölümleri çıkar
        sections = {
            'general_overview': extract_section(response, 'general_overview'),
            'technical_analysis': extract_section(response, 'technical_analysis'),
            'scenarios': {
                'high_probability': extract_section(response, 'high_probability'),
                'medium_probability': extract_section(response, 'medium_probability'),
                'low_probability': extract_section(response, 'low_probability')
            },
            'key_levels': extract_section(response, 'key_levels')
        }
        
        # HTML formatında birleştir
        html_response = f"""
        <div class="analysis-container">
            <div class="general-overview">
                <h4>Genel Durum</h4>
                {sections['general_overview']}
            </div>
            
            <div class="technical-analysis">
                <h4>Teknik Analiz</h4>
                {sections['technical_analysis']}
            </div>
            
            <div class="scenarios">
                <h4>Olası Senaryolar</h4>
                <div class="high-probability">
                    <h5>Yüksek Olasılıklı Senaryo</h5>
                    {sections['scenarios']['high_probability']}
                </div>
                <div class="medium-probability">
                    <h5>Orta Olasılıklı Senaryo</h5>
                    {sections['scenarios']['medium_probability']}
                </div>
                <div class="low-probability">
                    <h5>Düşük Olasılıklı Senaryo</h5>
                    {sections['scenarios']['low_probability']}
                </div>
            </div>
            
            <div class="key-levels">
                <h4>Önemli Seviyeler</h4>
                {sections['key_levels']}
            </div>
        </div>
        """
        
        return html_response
        
    except Exception as e:
        logging.error(f"Gemini yanıtı parse hatası: {str(e)}")
        return "Analiz sonuçları işlenemedi."

def extract_section(text, tag):
    """HTML/XML etiketleri arasındaki içeriği çıkarır"""
    try:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            return ""
            
        content = text[start_idx + len(start_tag):end_idx].strip()
        return content
    except Exception:
        return ""

def generate_fallback_analysis(summary, price_history):
    """Yedek analiz üretir"""
    try:
        total_occurrences = summary.get('total_occurrences', 0)
        avg_next_day = summary.get('average_next_day', 0)
        avg_next_week = summary.get('average_next_week', 0)
        avg_next_month = summary.get('average_next_month', 0)

        analysis = f"""
        Genel Durum:
        - Benzer fiyat seviyesi {total_occurrences} kez görülmüş
        - Ortalama getiriler:
          * Ertesi gün: {avg_next_day:.2f}%
          * 1 hafta: {avg_next_week:.2f}%
          * 1 ay: {avg_next_month:.2f}%

        Olası Senaryolar:
        1. Yüksek Olasılıklı Senaryo:
        - {total_occurrences} benzer durumun {summary.get('positive_next_month', 0)} tanesinde yükseliş görülmüş
        - Ortalama {avg_next_month:.2f}% aylık değişim

        2. Orta Olasılıklı Senaryo:
        - Kısa vadede yatay seyir
        - Haftalık bazda {avg_next_week:.2f}% değişim beklentisi

        3. Düşük Olasılıklı Senaryo:
        - Trend değişimi
        - Volatilite artışı
        """

        return analysis

    except Exception as e:
        logging.error(f"Yedek analiz hatası: {str(e)}")
        return "Analiz üretilemedi."

@csrf_exempt
@require_http_methods(["GET"])
def analyze_stock(request):
    try:
        symbol = request.GET.get('symbol', '').upper().strip()
        if not symbol:
            return JsonResponse({'error': 'Hisse senedi sembolü gereklidir'}, status=400)

        # Rate limit kontrolü
        ip = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or request.META.get('REMOTE_ADDR')
        rate_key = USER_RATE_LIMIT_KEY.format(ip=ip)
        
        if cache.get(rate_key):
            return JsonResponse({'error': 'Çok fazla istek atıyorsunuz, dakikada 2 istek hakkınız var.'}, status=429)
        
        cache.set(rate_key, 5, timeout=USER_RATE_LIMIT_SECONDS)

        # Veriyi al
        data = get_stock_data(symbol)
        
        # NaN değerleri None ile değiştir
        def clean_nan(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            return obj

        # Veriyi temizle
        data = clean_nan(data)

        # Gemini analizi ekle
        if 'summary' in data and 'price_history' in data and data['summary']['total_occurrences'] > 0:
            try:
                gemini_comment = gemini_flash_comment(data['summary'], data['technical_indicators'], data['price_history'])
                data['gemini_comment'] = gemini_comment
            except Exception as e:
                logging.error(f"Gemini yorum hatası: {str(e)}")
                data['gemini_comment'] = "Analiz üretilemedi."

        return JsonResponse(data, safe=False)

    except Exception as e:
        logging.error(f"Analiz hatası: {str(e)}", exc_info=True)
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen daha sonra tekrar deneyiniz.'}, status=500)

def home(request):
    return render(request, 'stock_analyzer/home.html')
