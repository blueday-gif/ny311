"""
[읽을거리] 뉴욕 311, 질문 100개 — 책 HTML 만들기

입력 : docs/hundred/questions.json (질문 설계), reports/book/hundred_numbers.json (analysis/book_hundred_questions.py 가 계산한 수치)
       outputs/figures/h100_Q001.png ~ h100_Q100.png
출력 : reports/book/nyc311_hundred_questions.html
순서 : analysis/book_hundred_questions.py → 이 파일 → reports/book/build_pdf_hundred.ps1
실행 : D:/dev/python/python.exe reports/book/build_hundred_html.py
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = json.load(open(ROOT / "docs" / "hundred" / "questions.json", encoding="utf-8"))
with open(ROOT / "reports" / "book" / "hundred_numbers.json", encoding="utf-8") as fh:
    V = json.load(fh)
assert not V["_check_diffs"], V["_check_diffs"]


def fx(v):
    """배수 표기: 10배 이상은 정수, 10배 미만은 소수 한 자리."""
    return f"{v:.0f}배" if v >= 10 else f"{v:.1f}배"


SA_NOTE = "브롱크스 북부의 한 주소(8장)는 빼고 셌습니다."
# 책에 쓸 문장 다듬기: q 질문, b 큰 숫자, l 큰 숫자 설명, a 한 줄 답, c 주의점(None 이면 없음)
O = {
    "Q001": dict(c="첫날과 마지막 날은 밤 9시 무렵에서 잘려 있어, 하루 단위 비교(평균·최대·최소)에서는 이 이틀을 뺐습니다."),
    "Q002": dict(a="평균 7.9초마다 한 건, 1분에 7.6건꼴이다. 오전 10시에는 1분에 11건까지 오른다."),
    "Q005": dict(c=None),
    "Q006": dict(a="2026년 2월 24일 폭설일. 하루 2만2,793건으로 평소의 2배, 그중 절반이 눈·얼음 신고였다."),
    "Q007": dict(l="크리스마스 ÷ 평소 하루"),
    "Q008": dict(b=fx(V["Q008"]["tue_over_sat"]), a="화요일 하루 1만1,700건, 토요일 9,847건. 평일이 주말보다 바쁘다."),
    "Q009": dict(l="7월 4일 ÷ 하루 평균"),
    "Q012": dict(a="이웃 소음. 평소 겨울 목요일보다 1.6배 많았다.",
                 c="성탄 이브에서 넘어온 성탄 0시대가 그날 이웃 소음의 정점입니다. 계절 효과가 섞이지 않게 같은 겨울(11~1월)의 다른 목요일 11일과 비교했고, " + SA_NOTE),
    "Q013": dict(c="성탄절은 청소국 수거가 쉬는 날이라 '미수거'라는 일 자체가 생기지 않는 것으로 보입니다. 눈 민원은 그날 날씨 탓이라 비교에서 뺐습니다."),
    "Q014": dict(l="새해 0시 ÷ 평소 0시", a="1월 1일 0시대 94건. 평소 0시대 평균 3.7건의 25배다."),
    "Q015": dict(l="평소 169건 → 1건", c="남은 1건은 시스템 기록 오류로 보입니다. 서머타임이 끝나는 가을 날(11월 1일)은 데이터가 끝난 뒤라 볼 수 없습니다."),
    "Q016": dict(c="브롱크스 북부의 한 주소를 빼지 않으면 2025년 11~12월 날짜들이 상위를 휩쓸어 버립니다(8장에서 다시 봅니다). 6월 21일은 일요일이라 토요일 밤에서 이어진 새벽 신고가 들어 있습니다."),
    "Q017": dict(l="25도 이상 ÷ 영하"),
    "Q018": dict(b=fx(V["Q018"]["ratio_25_vs_0"]), l="25도 이상 ÷ 영하", a="거의 안 탄다. 25도 넘는 날과 영하인 날의 차이가 1.03배뿐이다.",
                 c="한 주소(8장)를 빼지 않으면 겨울 값이 크게 부풀어 '추울수록 많다'로 잘못 보입니다."),
    "Q019": dict(b=fx(V["Q019"]["ratio_25_vs_0"]), l="25도 이상 ÷ 영하",
                 a=f"25도 넘는 날 하루 {V['Q019']['by_bin']['25+']:.0f}건, 영하인 날의 {fx(V['Q019']['ratio_25_vs_0'])}.",
                 c="겨울 값(하루 33건)에는 동파·고장 누수가 섞여 있습니다. 선선한 날(10~20도)과 비교하면 4.4배입니다."),
    "Q020": dict(l="영하 ÷ 20~25도", c="정전은 양 끝이 함께 높은 U자라, 이 질문만 '영하 대비'가 아니라 가운데 구간(20~25도)과 비교했습니다."),
    "Q021": dict(l="25도 이상 ÷ 영하", a="25도 넘는 날 하루 20건, 영하인 날의 4.5배. 길거리 소음 다음으로 더위를 탄다.",
                 c="그림은 영하인 날에도 하루 4건 이상 들어오는 민원만 비교했습니다. 불꽃놀이처럼 한철 행사인 민원은 뺐습니다."),
    "Q022": dict(l="25도 이상 ÷ 영하"),
    "Q023": dict(l="2월 23~24일 비중"),
    "Q024": dict(l="스태튼아일랜드 ÷ 맨해튼 (1인당)", a="스태튼아일랜드. 10만 명당 3,072건으로, 가장 적은 맨해튼(280건)의 11배다."),
    "Q025": dict(a="폭설 첫날 2월 23일. 하루 1,496건, 평소의 12배 나무가 꺾였다."),
    "Q026": dict(b=fx(V["Q026"]["mar_over_month_avg"]), l="3월 ÷ 월평균",
                 a=f"3월. 2만2,761건으로 한 달 평균의 {fx(V['Q026']['mar_over_month_avg'])}다.",
                 c="얼었다 녹기를 되풀이한 늦겨울 뒤에 도로가 한꺼번에 깨지는 것으로 보입니다."),
    "Q027": dict(c="한 주소(8장)를 넣으면 음악·파티 비중이 55%로 달라집니다."),
    "Q028": dict(a="오전 9시가 정점. 밤 10시에 몰리는 파티 소음과 달리 낮 민원이다."),
    "Q031": dict(l="맨해튼 ÷ 브롱크스 (1인당)", a="62배. 맨해튼 하늘의 헬기 소리가 압도적이다.",
                 c="소수 주소의 반복 신고 영향이 큽니다. 상위 10개 주소가 38%를 냅니다(Q073)."),
    "Q032": dict(l="맨해튼 7번 커뮤니티보드 비중"),
    "Q033": dict(a="허가 시간 밖 공사가 25%로 1위다.", c="'공사(옛 분류)' 7,575건은 다른 분류 체계로 들어온 일반 공사 소음이라 따로 셌습니다."),
    "Q034": dict(c="요일이 고르게 퍼져 있다면 토·일요일 몫은 7일 중 2일, 28.6%입니다."),
    "Q036": dict(l="스태튼아일랜드 ÷ 맨해튼 (1인당)", a="스태튼아일랜드. 길에서 발견되는 너구리가 1인당 맨해튼의 23배다."),
    "Q037": dict(l="스태튼아일랜드 비중"),
    "Q039": dict(c="등딱지 10cm 미만 거북은 살모넬라균 위험 때문에 판매가 금지돼 있습니다."),
    "Q040": dict(l="맨해튼 ÷ 스태튼아일랜드 (1인당)"),
    "Q044": dict(l="맨해튼 ÷ 브롱크스 (1인당)", a="비둘기 배설물 신고, 맨해튼은 1인당 브롱크스의 6.5배다."),
    "Q048": dict(b=fx(V["Q048"]["QN_over_MN"]), l="퀸스 ÷ 맨해튼 (1인당)",
                 a="퀸스. 1인당 맨해튼의 12배. 차고 딸린 집이 많은 곳에서 신고가 나온다.",
                 c="그림은 6장의 도로 민원 7종을 한 장에 모은 것입니다. 칸의 숫자는 각 민원의 뉴욕 평균을 1로 놓은 1인당 배수입니다."),
    "Q049": dict(l="퀸스 ÷ 맨해튼 (1인당)", a="퀸스. 10만 명당 1,287건, 맨해튼의 7.8배다."),
    "Q050": dict(b=fx(V["Q050"]["MN_over_SI"]), l="맨해튼 ÷ 스태튼아일랜드 (1인당)",
                 a=f"맨해튼·브루클린. 1인당 가장 적은 스태튼아일랜드의 {fx(V['Q050']['MN_over_SI'])}, 퀸스의 3.9배다."),
    "Q051": dict(l="스태튼아일랜드 ÷ 브루클린 (1인당)", a="스태튼아일랜드. 가장 적은 브루클린의 3.2배, 밤길이 어두운 곳이 많다는 뜻으로 보인다."),
    "Q052": dict(l="브루클린 ÷ 브롱크스 (1인당)", c="1인당 가장 적은 곳은 스태튼아일랜드지만 1년에 9건뿐이라 비교가 불안정해, 그다음으로 적은 브롱크스와 비교했습니다."),
    "Q053": dict(l="퀸스+브롱크스 비중", c="공유 전동킥보드가 퀸스·브롱크스 일부 지역에서만 운영되기 때문으로 보입니다(데이터 밖의 사실)."),
    "Q054": dict(l="맨해튼 비중", a=f"42%. 1인당으로는 가장 적은 스태튼아일랜드의 {fx(V['Q054']['MN_over_SI'])}다."),
    "Q055": dict(l="오전 8시 ÷ 새벽 3시"),
    "Q056": dict(l="브롱크스 ÷ 스태튼아일랜드 (1인당)", a="브롱크스. 10만 명당 2만 건이 넘는다. 스태튼아일랜드의 7.4배.",
                 c="주택 수리 민원은 세입자가 내므로 자치구마다 임대주택이 얼마나 많은지가 크게 반영됩니다."),
    "Q059": dict(c="같은 건물의 여러 세대가 같은 날 신고한 경우도 각각 센 '신고 수' 기준입니다."),
    "Q060": dict(a="브롱크스 북부의 한 건물. 1년에 2,211건, 138일 동안 신고가 들어왔다.",
                 c="개인 거주지일 수 있어 1위 건물은 동네만 적었습니다."),
    "Q061": dict(b=fx(V["Q061"]["BX_over_SI"]), l="브롱크스 ÷ 스태튼아일랜드 (1인당)",
                 a=f"브롱크스. 1인당 스태튼아일랜드의 {fx(V['Q061']['BX_over_SI'])}, 퀸스와 비교해도 4.5배다.",
                 c="스태튼아일랜드는 엘리베이터가 있는 고층 건물이 적은 것으로 보입니다."),
    "Q063": dict(c="그림은 1년에 1만 건 넘게 들어온 주택 민원 중 많은 10종입니다."),
    "Q064": dict(c="난방 민원의 30.7%는 '이미 다른 세입자가 신고한 건'으로 닫힙니다(Q088). 그림은 많은 10종입니다."),
    "Q065": dict(a="브롱크스 북부의 한 주소. 1년에 4만7,105건, 뉴욕 전체 신고의 1.2%다.",
                 c="개인 거주지일 수 있어 동네만 적었습니다. 이 주소의 신고는 거의 전부(4만7,013건)가 이웃 소음입니다."),
    "Q066": dict(l="뉴욕 이웃 소음 중 비중"),
    "Q067": dict(l="그 주소 포함 ÷ 제외"),
    "Q068": dict(a="2025년 11월 30일 하루 2,195건, 약 40초에 한 번꼴. 24시간 내내 쉬지 않았다.",
                 c="이 주소의 1년 치 신고 90%가 11월 7일~12월 16일, 약 6주 사이에 몰려 있습니다."),
    "Q069": dict(a="63%가 '가 보니 흔적 없음'. 소환장(딱지)은 4만7천 건 중 10건뿐이었다.",
                 l="'흔적 없음' 비율", c="이 주소 신고의 99.8%는 앱으로 들어왔습니다."),
    "Q070": dict(a="브루클린 남부의 한 주소. 데이터 속 365일 중 364일 신고했다, 하루 24건꼴.",
                 c="개인 거주지일 수 있어 동네만 적었습니다. 이 주소 신고의 99.8%가 이웃 소음입니다."),
    "Q071": dict(a="퀸스 서부의 한 주소. 7,604건, 신고한 날마다 22건꼴이다.",
                 c="개인 주소일 수 있어 동네만 적었습니다. 96.5%가 '보도를 막은 차' 신고입니다."),
    "Q072": dict(a="62%가 퀸스 파슨스 대로의 한 주소에서 나왔다.", c="번지는 적지 않았습니다."),
    "Q073": dict(c=None),
    "Q074": dict(c="주소 칸에 거리 이름만 적힌 경우(예: 'BROADWAY')도 한 곳으로 세었기 때문에 '주소'는 근사치입니다."),
    "Q075": dict(l="반복 신고 묶음에 든 비율"),
    "Q076": dict(c=SA_NOTE),
    "Q077": dict(l="이웃 소음 소환장 발부율"),
    "Q078": dict(l="차고 앞 막힘 발부율"),
    "Q079": dict(a="경찰이 가 보면 73%는 이미 끝나 있었다."),
    "Q080": dict(b="1,163건", l="1,400건에 1건", a="1,163건, 0.07%. 체포로 끝난 신고는 길거리 소음이 가장 많았다."),
    "Q082": dict(a="36%만 텐트를 확인했다. 46%는 가 보니 텐트가 없었다.",
                 c="노숙 텐트 신고는 공원국 등 다른 기관도 받기 때문에, 여기서는 경찰이 받은 것만 셌습니다."),
    "Q084": dict(l="브롱크스 ÷ 스태튼아일랜드"),
    "Q085": dict(c="실제 처리 시각이 없는 기록 11.9만 건은 뺐습니다. 경찰 민원끼리만 비교했습니다."),
    "Q086": dict(c="'찾지 못함', '증거 없음', '이미 가고 없음' 같은 문장이 들어 있는 답변을 셌습니다."),
    "Q087": dict(a="14%. 지원팀이 도움을 제안해도 받지 않거나 거부했다.", b="14%",
                 c="'도움을 받지 않음'(11.1%)과 '서비스를 거부함'(2.6%)을 합친 값입니다. 절반 가까이(48%)는 가 보니 사람을 찾지 못했습니다."),
    "Q089": dict(c="공원국 답변 전체의 9.2%입니다."),
    "Q090": dict(a="1년에 2만 번 넘게(2만186번). 83%가 6~8월이다."),
    "Q091": dict(c="실제 처리 시각이 없는 기록 11.9만 건은 뺐습니다."),
    "Q092": dict(c="공사 납 먼지 민원(1,316건)도 100% 미종료입니다. 둘 다 접수만 해 두는 기록용으로 쓰이는 것으로 보입니다."),
    "Q094": dict(c=SA_NOTE + " 이 주소를 넣으면 앱 비중이 22.4%로 올라갑니다."),
    "Q095": dict(a="구걸(팬핸들링). 91%가 앱으로 들어온다.",
                 c="헬기·불꽃놀이처럼 앱에 메뉴가 없는 민원은 앱 비율이 0%입니다. 방치 차량은 100% 전화로 기록돼 있어 그림에서 뺐습니다."),
    "Q096": dict(l="스태튼아일랜드 전화 비중"),
    "Q098": dict(a="가방·지갑이 49%. 그리고 98%가 택시에서 잃어버린 물건이다."),
    "Q100": dict(q="1년에 10건 이하로 들어온 민원은?",
                 c="그림에는 10건 이하 민원 일부와 그보다 조금 많은 별난 민원(방사성 물질 15건, 인명구조원 30건, 공중화장실 66건)을 함께 놓았습니다."),
}

# 장마다: 여는 쪽 안내 / 가장 놀라운 숫자 / (혼자 남은 쪽에 넣을) 이 장의 숫자 3개
CH_NOTE = {3: "이 장의 날씨 비교는 브롱크스 북부의 한 주소를 빼고, 기록이 온전한 363일로 셌습니다. 배수는 모두 '영하인 날'과 비교했습니다.",
           4: "이 장은 브롱크스 북부의 한 주소를 빼고 셌습니다. 이유는 8장에서 봅니다.",
           6: "자치구는 인구 10만 명당 건수로 비교하고, '몇 배'는 가장 낮은 구와 견줬습니다.",
           8: "개인 거주지일 수 있는 주소는 동네만 적고, 기관(종교시설·공항 등)은 거리 이름까지 적었습니다.",
           9: "이 장은 브롱크스 북부의 한 주소를 빼고 셌습니다. 경찰이 남긴 결과 문장을 읽어 하나씩 분류했습니다.",
           11: "이 장은 브롱크스 북부의 한 주소를 빼고 셌습니다."}
CH_HERO = {1: ("396만 건", "1년 동안 311에 들어온 신고"), 2: ("75배", "7월 4일 불꽃놀이 신고, 평소 하루 대비"),
           3: ("5.7배", "길거리 소음, 25도 이상인 날 ÷ 영하인 날"), 4: ("62배", "헬기 소음, 맨해튼 ÷ 브롱크스 (1인당)"),
           5: ("79%", "사슴 사체 신고 중 스태튼아일랜드 몫"), 6: ("12배", "차고 앞 막힘, 퀸스 ÷ 맨해튼 (1인당)"),
           7: ("27%", "검사관이 집에 들어가지 못한 주택 민원"), 8: ("4만7,105건", "한 주소가 1년에 낸 신고"),
           9: ("0.4%", "이웃 소음 신고에 딱지가 나온 비율"), 10: ("48%", "노숙인 지원팀이 가 보니 사람이 없던 비율"),
           11: ("91%", "구걸 신고 중 앱으로 들어온 몫"), 12: ("93 : 1", "택시 불만 대 칭찬")}
CH_SUM = {2: [("75배", "7월 4일 불꽃놀이 신고 ÷ 평소 하루"), ("1건", "서머타임 시작일 새벽 2시대 신고 (평소 169건)"), ("1.6배", "성탄절 이웃 소음 ÷ 평소 겨울 목요일")],
          6: [("28%", "불법주차 신고 중 '소화전 앞'"), ("12배", "차고 앞 막힘, 퀸스 ÷ 맨해튼 (1인당)"), ("94%", "전동킥보드 신고 중 퀸스·브롱크스 몫")],
          7: [("7.4배", "주택 수리 민원, 브롱크스 ÷ 스태튼아일랜드 (1인당)"), ("23%", "상위 1% 건물이 내는 난방 민원 몫"), ("4%", "난방 민원에 위반 딱지가 붙는 비율")],
          11: [("54%", "웹(온라인) 신고 비중"), ("91%", "구걸 신고 중 앱 몫"), ("30%", "스태튼아일랜드의 전화 신고 비중")],
          12: [("93 : 1", "택시 불만 대 칭찬"), ("98%", "분실물 중 택시에서 잃어버린 것"), ("1건", "1년 동안 '비둘기 포획' 신고")]}
HEROES = {"Q009", "Q065", "Q097"}  # 한 쪽 전체를 쓰는 질문

parts = []
for q in Q:
    o = O.get(q["id"], {})
    for k_src, k in [("question", "q"), ("big_number", "b"), ("big_label", "l"), ("answer", "a"), ("caveat", "c")]:
        q[k] = o[k] if k in o else q.get(k_src)
    if not parts or parts[-1]["name"] != q["part"]:
        parts.append({"name": q["part"], "intro": q["part_intro"], "qs": []})
    parts[-1]["qs"].append(q)
assert len(parts) == 12 and sum(len(p["qs"]) for p in parts) == 100

# 코드·컬럼명·파일 경로·번지가 새어 나가지 않았는지 검사
BAD = re.compile(r"address_key|flag_|problem|value_counts|category|resolution|_pct|\bSA\b|\[B\]|FULL|POP|\.py|\.json|\.csv|analysis/|outputs/")
for q in Q:
    for k in "qblac":
        t = q[k] or ""
        assert not BAD.search(t.replace("÷", "")), (q["id"], k, t)
        assert not re.search(r"\d+[-A-Z]?\d* (EAST|WEST|STREET|AVENUE|PARSONS)", t), q["id"]
    if q["id"] in ("Q060", "Q065", "Q070", "Q071"):
        assert "번가" not in q["a"] + (q["c"] or "") and "애비뉴" not in q["a"], q["id"]


def e(s):
    return html.escape(s or "")


def bignum(s):
    m = re.match(r"^([0-9][0-9,.: ]*(?:만[0-9,]*)?)(.*)$", s)
    if m and m.group(1).strip():
        return f'<span class="bn">{e(m.group(1).strip())}</span><span class="bu">{e(m.group(2))}</span>'
    return f'<span class="bn bw">{e(s)}</span>'


def img(qid):
    return f'<img src="../../outputs/figures/h100_{qid}.png" alt="">'


def block(q):
    cav = f'<p class="cav">{e(q["c"])}</p>' if q["c"] else ""
    return f'''<article class="qb">
  <header><span class="qn">{q["id"]}</span><h3>{e(q["q"])}</h3></header>
  <div class="body">
    <div class="big"><div class="num">{bignum(q["b"])}</div><div class="lab">{e(q["l"])}</div></div>
    <figure>{img(q["id"])}</figure>
  </div>
  <p class="ans">{e(q["a"])}</p>
  {cav}
</article>'''


def hero(q):
    cav = f'<p class="cav">{e(q["c"])}</p>' if q["c"] else ""
    return f'''<article class="hero">
  <header><span class="qn">{q["id"]}</span><h3>{e(q["q"])}</h3></header>
  <div class="hrow"><div class="num">{bignum(q["b"])}</div><div><div class="lab">{e(q["l"])}</div><p class="ans">{e(q["a"])}</p></div></div>
  <figure>{img(q["id"])}</figure>
  {cav}
</article>'''


def sumbox(i):
    items = CH_SUM.get(i)
    if not items:
        return ""
    cells = "".join(f'<div class="cell"><div class="sn">{e(n)}</div><div class="sl">{e(l)}</div></div>' for n, l in items)
    return f'<aside class="sumbox"><div class="tag">이 장의 숫자 3개</div><div class="cells">{cells}</div></aside>'


CSS = r"""
@page { size: A4; margin: 13mm 16mm 15mm 16mm;
  @bottom-center { content: counter(page); font: 8pt "Malgun Gothic"; color: #9a9384; letter-spacing:.1em; } }
@page :first { margin: 0; @bottom-center { content: none; } }
:root { --ink:#1d2433; --muted:#5d6678; --accent:#C2410C; --note-bg:#F4F1EA; --note-bd:#D8CFBD; --rule:#E3E0D8; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin:0; font-family:"Malgun Gothic", sans-serif; color:var(--ink); font-size:10pt; line-height:1.6; background:#fff; word-break:keep-all; }
h1,h2,h3 { line-height:1.3; margin:0; }
p { margin:0 0 .7em; }
b, strong { color:#000; }
.pg { break-after:page; }
.pg.h { height:262mm; display:flex; flex-direction:column; justify-content:center; }
/* 표지 */
.cover { width:210mm; height:297mm; overflow:hidden; position:relative; display:flex; flex-direction:column; justify-content:space-between; break-after:page;
  background: linear-gradient(165deg,#121826 0%,#1E2A44 55%,#3A2622 100%); color:#F5EFE3; padding:34mm 24mm 22mm; }
.cover::before { content:""; position:absolute; left:24mm; top:24mm; width:18mm; height:1.2mm; background:#F2A65A; }
.cover .kicker { letter-spacing:.35em; font-size:8.5pt; color:#E9B872; }
.cover h1 { font-size:40pt; font-weight:800; margin-top:12mm; line-height:1.22; letter-spacing:-.01em; }
.cover h1 span { color:#F2A65A; }
.cover .sub { font-size:12.5pt; margin-top:8mm; color:#D9D2C3; line-height:1.75; }
.cover .hundred { font-size:120pt; font-weight:800; color:rgba(242,166,90,.10); position:absolute; right:16mm; top:92mm; line-height:1; letter-spacing:-.04em; }
.cover .qs { font-size:9.6pt; color:#D6CEBD; border-top:1px solid rgba(233,184,114,.45); padding-top:6mm; display:grid; grid-template-columns:1fr 1fr; column-gap:10mm; }
.cover .qs div { margin:1.6mm 0; }
.cover .qs b { color:#F2A65A; font-weight:700; display:inline-block; width:8mm; }
.cover .meta { font-size:7.8pt; color:#97A0B4; line-height:1.7; border-top:1px solid rgba(255,255,255,.12); padding-top:4mm; }
/* 차례 */
.toc { break-after:page; padding-top:4mm; }
.toc h2 { font-size:11pt; letter-spacing:.4em; color:var(--accent); margin-bottom:6mm; padding-bottom:3mm; border-bottom:2px solid var(--ink); }
.toc ol { list-style:none; padding:0; margin:0; }
.toc ol > li { display:grid; grid-template-columns: 16mm 1fr 24mm; padding:1.8mm 0; border-bottom:1px solid var(--rule); align-items:baseline; }
.toc .n { font-size:15pt; font-weight:800; color:var(--accent); line-height:1; }
.toc .t { font-size:11pt; font-weight:700; line-height:1.4; }
.toc .s { font-size:8pt; color:var(--muted); margin-top:.2mm; line-height:1.4; }
.toc .r { font-size:8.4pt; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
.howto { break-inside:avoid; margin-top:6mm; background:var(--note-bg); border:1px solid var(--note-bd); border-radius:2mm; padding:3.6mm 5mm; font-size:8.8pt; line-height:1.65; display:grid; grid-template-columns: 46mm 1fr; column-gap:6mm; align-items:center; }
.howto .demo { border-top:2px solid var(--ink); padding-top:1.5mm; }
.howto .demo .ex { font-size:7pt; letter-spacing:.2em; color:#8a6d3b; font-weight:700; margin-left:2mm; }
.howto .demo .bn { font-size:22pt; }
.howto ul { margin:0; padding-left:4.5mm; }
.howto li { margin:.5mm 0; }
.tag { display:inline-block; font-size:6.8pt; letter-spacing:.18em; color:#fff; background:#8a6d3b; padding:.3mm 2mm; border-radius:1mm; margin-bottom:1.5mm; }
/* 장 여는 쪽 */
.opener { break-before:page; break-after:page; height:262mm; display:flex; flex-direction:column; justify-content:center; padding:0 6mm; }
.opener .no { font-size:9pt; letter-spacing:.35em; color:var(--accent); font-weight:700; }
.opener .chn { font-size:54pt; font-weight:800; color:var(--accent); line-height:1; margin:2mm 0 4mm; letter-spacing:-.03em; }
.opener h2 { font-size:28pt; font-weight:800; letter-spacing:-.01em; padding-bottom:5mm; border-bottom:2px solid var(--ink); }
.opener .intro { font-size:12pt; color:#333; margin-top:6mm; line-height:1.75; max-width:150mm; }
.opener .wow { margin-top:9mm; display:flex; align-items:baseline; gap:6mm; background:var(--note-bg); border:1px solid var(--note-bd); border-radius:2mm; padding:5mm 7mm; }
.opener .wow .k { font-size:7.5pt; letter-spacing:.25em; color:#8a6d3b; font-weight:700; writing-mode:horizontal-tb; flex:none; }
.opener .wow .v { font-size:44pt; font-weight:800; color:var(--accent); line-height:1; letter-spacing:-.02em; flex:none; }
.opener .wow .d { font-size:10.5pt; color:#333; line-height:1.5; }
.opener .note { font-size:8.6pt; color:#8a6d3b; margin-top:4mm; }
.opener ol { list-style:none; padding:0; margin:9mm 0 0; columns:2; column-gap:10mm; font-size:9pt; color:#444; }
.opener ol li { break-inside:avoid; padding:1.3mm 0; border-bottom:1px solid var(--rule); line-height:1.45; }
.opener ol li span { color:var(--accent); font-weight:700; font-size:8pt; margin-right:2mm; font-variant-numeric:tabular-nums; }
/* 질문 블록: 한 쪽에 두 개 */
.qb { break-inside:avoid; border-top:2px solid var(--ink); padding-top:2.6mm; margin:0 0 7mm; }
.qb header, .hero header { display:flex; align-items:baseline; gap:3.5mm; margin-bottom:2.6mm; }
.qn { font-size:8.5pt; font-weight:800; color:#fff; background:var(--accent); padding:.4mm 2mm; border-radius:1mm; letter-spacing:.06em; font-variant-numeric:tabular-nums; flex:none; position:relative; top:-.5mm; }
.qb h3 { font-size:13.5pt; font-weight:800; letter-spacing:-.01em; line-height:1.35; }
.qb .body { display:grid; grid-template-columns: 45mm 127mm; column-gap:6mm; align-items:start; }
.big { padding-top:2mm; }
.num { line-height:1.05; color:var(--accent); }
.bn { font-size:30pt; font-weight:800; letter-spacing:-.02em; color:var(--accent); }
.bn.bw { font-size:24pt; }
.bu { font-size:15pt; font-weight:800; margin-left:.6mm; color:var(--accent); }
.lab { font-size:8.6pt; color:var(--muted); line-height:1.45; }
.big .lab { margin-top:2.2mm; border-top:1px solid var(--rule); padding-top:2mm; }
.qb figure, .hero figure { margin:0; }
.qb figure img { width:127mm; display:block; }
.ans { font-size:10.6pt; font-weight:700; margin:2.6mm 0 0; padding-left:5mm; position:relative; line-height:1.55; }
.ans::before { content:""; position:absolute; left:0; top:.55em; width:2.4mm; height:2.4mm; background:var(--accent); }
.cav { font-size:8pt; color:var(--muted); margin:1.4mm 0 0 5mm; line-height:1.5; }
.cav::before { content:"주의 "; color:#8a6d3b; font-weight:700; letter-spacing:.1em; }
/* 한 쪽 전체를 쓰는 질문 */
.hero { border-top:3px solid var(--ink); padding-top:5mm; }
.hero h3 { font-size:20pt; font-weight:800; letter-spacing:-.01em; }
.hero .hrow { display:grid; grid-template-columns: auto 1fr; column-gap:9mm; align-items:center; margin:6mm 0 7mm; padding:5mm 0; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }
.hero .bn { font-size:58pt; }
.hero .bn.bw { font-size:44pt; }
.hero .bu { font-size:26pt; }
.hero .lab { font-size:10pt; }
.hero .ans { font-size:13pt; margin-top:2mm; }
.hero figure img { width:178mm; display:block; }
.hero .cav { font-size:8.6pt; margin-top:3mm; }
/* 혼자 남은 쪽의 요약 상자 */
.sumbox { break-inside:avoid; margin-top:10mm; background:var(--note-bg); border:1px solid var(--note-bd); border-radius:2mm; padding:5mm 7mm 6mm; }
.sumbox .cells { display:grid; grid-template-columns:1fr 1fr 1fr; column-gap:7mm; margin-top:3mm; }
.sumbox .cell { border-top:2px solid var(--ink); padding-top:3mm; }
.sumbox .sn { font-size:32pt; font-weight:800; color:var(--accent); line-height:1.05; letter-spacing:-.02em; }
.sumbox .sl { font-size:9pt; color:#3b2f1a; margin-top:2mm; line-height:1.5; }
/* 맺음말 · 방법 메모 */
.epi, .method { break-before:page; }
.epi h2, .method h2 { font-size:21pt; font-weight:800; margin-bottom:6mm; letter-spacing:-.01em; padding-bottom:3mm; border-bottom:2px solid var(--ink); }
.epi .lead { font-size:11pt; line-height:1.8; margin-bottom:6mm; }
.epi ol { padding:0; margin:0; list-style:none; counter-reset:k; }
.epi ol li { counter-increment:k; display:grid; grid-template-columns:10mm 1fr; padding:3.2mm 0; border-bottom:1px solid var(--rule); font-size:10pt; line-height:1.7; }
.epi ol li::before { content:counter(k); font-size:16pt; font-weight:800; color:var(--accent); line-height:1.1; }
.epi .rest { margin-top:7mm; background:#1d2433; color:#F5EFE3; border-radius:2mm; padding:4.5mm 6mm; font-size:9.6pt; line-height:1.75; }
.epi .rest b { color:#F2A65A; }
.epi .rest .lab { font-size:7.5pt; letter-spacing:.3em; color:#E9B872; display:block; margin-bottom:1.5mm; }
.method dl { margin:0; display:grid; grid-template-columns: 34mm 1fr; column-gap:6mm; }
.method dt { font-weight:800; font-size:9.4pt; padding:2.6mm 0; border-bottom:1px solid var(--rule); color:var(--accent); }
.method dd { margin:0; font-size:9pt; line-height:1.65; padding:2.6mm 0; border-bottom:1px solid var(--rule); }
.method table { border-collapse:collapse; font-size:8.4pt; margin:2mm 0 .5mm; width:100%; table-layout:fixed; }
.method td, .method th { padding:.8mm 1mm; text-align:center; border:0; font-variant-numeric:tabular-nums; }
.method th { font-weight:700; color:#333; border-bottom:1px solid var(--rule); }
.method .src { font-size:7.8pt; color:var(--muted); margin-top:5mm; line-height:1.6; }
"""

out = [f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>뉴욕 311, 질문 100개</title>
<style>{CSS}</style>
</head>
<body>

<!-- ───────────── 표지 ───────────── -->
<section class="cover">
  <div>
    <div class="kicker">NYC 311 · 100 QUESTIONS</div>
    <h1>뉴욕 311,<br><span>질문 100개</span></h1>
    <div class="sub">396만 건의 민원에 던진 백 개의 궁금증<br>— 답은 숫자 하나, 그림 하나</div>
  </div>
  <div class="hundred">100</div>
  <div class="qs">
"""]
for i, p in enumerate(parts, 1):
    out.append(f'    <div><b>{i}</b>{e(p["name"])}</div>\n')
out.append("""  </div>
  <div class="meta">자료: NYC Open Data 311 Service Requests (2025-11-01 ~ 2026-10-31, 전처리본) · 기온: Open-Meteo 과거 기상 자료 · 인구: 2020 미국 센서스</div>
</section>

<!-- ───────────── 차례 ───────────── -->
<section class="toc">
  <h2>차례</h2>
  <ol>
""")
for i, p in enumerate(parts, 1):
    a, b = p["qs"][0]["id"], p["qs"][-1]["id"]
    out.append(f'    <li><span class="n">{i}</span><div><div class="t">{e(p["name"])}</div><div class="s">{e(p["intro"])}</div></div><span class="r">{a} – {b}</span></li>\n')
out.append("""    <li><span class="n">&#9670;</span><div><div class="t">맺음말 · 데이터와 방법 메모</div><div class="s">100개의 답에서 보인 것, 그리고 숫자를 만든 규칙</div></div><span class="r"></span></li>
  </ol>
  <div class="howto">
    <div class="demo"><span class="qn">Q002</span><span class="ex">예시</span><div style="margin-top:2mm"><span class="bn">7.9</span><span class="bu">초</span></div><div class="lab" style="font-size:8pt">신고 1건당 간격 — 2번 질문의 큰 숫자</div></div>
    <div><span class="tag">읽는 법</span>
      <ul>
        <li>한 쪽에 질문 두 개. 질문마다 <b>큰 숫자 하나</b>, <b>그림 하나</b>, <b>한 줄 답</b>이 있습니다. 몇몇 질문은 한 쪽 전체를 씁니다.</li>
        <li>그림에서 <b style="color:#C2410C">주황색</b>이 답이 가리키는 곳입니다. 나머지는 비교를 위한 회색입니다.</li>
        <li>작은 글씨 <b style="color:#8a6d3b">주의</b>는 숫자를 읽을 때 알아 둘 점입니다. 숫자를 만든 규칙은 책 끝 '데이터와 방법 메모'에 모았습니다.</li>
      </ul>
    </div>
  </div>
</section>
""")

n_pages_q = 0
for i, p in enumerate(parts, 1):
    note = f'<div class="note">{e(CH_NOTE[i])}</div>' if i in CH_NOTE else ""
    hv, hl = CH_HERO[i]
    lis = "".join(f'<li><span>{q["id"]}</span>{e(q["q"])}</li>' for q in p["qs"])
    out.append(f"""
<!-- ───────────── {i}장 ───────────── -->
<section class="opener">
  <div class="no">CHAPTER {i} · 질문 {len(p["qs"])}개</div>
  <div class="chn">{i:02d}</div>
  <h2>{e(p["name"])}</h2>
  <div class="intro">{e(p["intro"])}</div>
  <div class="wow"><span class="k">이 장의<br>놀라운 숫자</span><span class="v">{e(hv)}</span><span class="d">{e(hl)}</span></div>
  {note}
  <ol>{lis}</ol>
</section>
""")
    # 쪽 나누기: 큰 질문은 한 쪽, 나머지는 두 개씩. 혼자 남은 쪽에는 '이 장의 숫자 3개'
    pages, buf = [], []
    for q in p["qs"]:
        if q["id"] in HEROES:
            if buf:
                pages.append(buf); buf = []
            pages.append([q])
        else:
            buf.append(q)
            if len(buf) == 2:
                pages.append(buf); buf = []
    if buf:
        pages.append(buf)
    used_sum = False
    for pg in pages:
        n_pages_q += 1
        if pg[0]["id"] in HEROES:
            out.append(f'<div class="pg h">{hero(pg[0])}</div>\n')
            continue
        extra = ""
        if len(pg) == 1 and not used_sum:
            extra = sumbox(i); used_sum = True
            assert extra, f"{i}장에 요약 상자 내용이 없음"
        out.append('<div class="pg">' + "\n".join(block(q) for q in pg) + extra + "</div>\n")

out.append("""
<!-- ───────────── 맺음말 ───────────── -->
<section class="epi">
  <h2>맺음말</h2>
  <p class="lead">질문 100개를 지나고 나면 311은 '불만 접수 창구'라기보다 도시가 매일 남기는 일기처럼 보입니다. 백 개의 답을 다섯 줄로 줄이면 이렇습니다.</p>
  <ol>
    <li><div><b>달력과 날씨가 민원을 움직인다.</b> 7월 4일 불꽃놀이 신고는 평소의 75배, 폭설 이틀에 눈 민원 1년 치의 28%가 몰렸습니다. 반면 이웃집 소음은 기온을 거의 타지 않았습니다.</div></li>
    <li><div><b>자치구마다 성격이 있다.</b> 차고 앞 막힘은 퀸스, 헬기 소음은 맨해튼, 집 수리 민원은 브롱크스, 너구리와 사슴은 스태튼아일랜드. 인구로 나누어야 비로소 보이는 차이입니다.</div></li>
    <li><div><b>몇몇 주소가 통계를 흔든다.</b> 한 주소가 뉴욕 이웃 소음의 11%를 냈고, 상위 1% 주소가 전체 신고의 30%를 냅니다. '많다'는 숫자를 보면 먼저 누가 냈는지를 물어야 합니다.</div></li>
    <li><div><b>신고 뒤의 결말은 대개 조용하다.</b> 경찰 출동의 41%는 '가 보니 흔적 없음'이었고, 이웃 소음에 딱지가 나온 것은 0.4%뿐입니다.</div></li>
    <li><div><b>신고하는 방법도 동네를 닮았다.</b> 웹이 절반을 넘지만, 스태튼아일랜드는 열 건 중 세 건을 아직 전화로 신고합니다.</div></li>
  </ol>
  <div class="rest"><span class="lab">이 책이 말해 주지 못하는 것</span>
    311은 <b>신고된 것</b>만 담습니다. 신고가 많다고 문제가 많은 것이 아니고, 신고가 적다고 조용한 것도 아닙니다. 신고를 잘 하는 동네와 사람이 따로 있을 수 있습니다.
    또 이 책의 비교는 대부분 <b>함께 움직인다</b>는 것까지만 보여 줍니다. 더운 날 길거리 소음이 는다는 것은 사실이지만, 더위 '때문'인지 사람이 밖에 많아서인지는 이 데이터만으로 가를 수 없습니다.
    1년 치 자료라서 해마다 되풀이되는 일인지도 아직 모릅니다.</div>
</section>

<!-- ───────────── 데이터와 방법 메모 ───────────── -->
<section class="method">
  <h2>데이터와 방법 메모</h2>
  <dl>
    <dt>무엇을 셌나</dt>
    <dd>뉴욕시 311에 2025년 11월 1일 밤 9시부터 2026년 10월 31일 밤 9시까지 1년 동안 접수된 민원입니다. 같은 신고가 시스템에서 두 번 기록된 줄은 먼저 지워 <b>3,962,167건</b>을 남겼고, 모든 질문이 여기서 출발합니다.</dd>
    <dt>하루 평균</dt>
    <dd>첫날과 마지막 날은 밤 9시 무렵에서 잘려 있어, 하루 평균·최다일·최소일은 이 이틀을 뺀 <b>363일</b>로 계산했습니다. 신고가 한 건도 없는 날은 0건으로 넣어 평균을 냈습니다.</dd>
    <dt>1인당 비교</dt>
    <dd>자치구끼리는 건수를 그대로 비교하지 않고 <b>인구 10만 명당 건수</b>로 바꿔 비교했고, '몇 배'는 원칙적으로 <b>가장 낮은 구와 견준 값</b>입니다(건수가 너무 적은 구를 피한 경우는 그 질문에 따로 적었습니다). 인구는 2020 센서스 값입니다.
      <table><tr><th>맨해튼</th><th>브루클린</th><th>퀸스</th><th>브롱크스</th><th>스태튼아일랜드</th></tr>
      <tr><td>169만</td><td>274만</td><td>241만</td><td>147만</td><td>50만</td></tr></table></dd>
    <dt>기온 구간</dt>
    <dd>맨해튼 부근의 하루 평균 기온으로 날을 다섯 무리로 나눴습니다. 영하 59일, 0~10도 90일, 10~20도 89일, 20~25도 86일, 25도 이상 39일. 구간마다 '하루 평균 신고 수'를 비교했으므로 날 수가 달라도 공정합니다. 배수는 '영하인 날'과 견줬습니다(양 끝이 함께 높은 정전만 예외).</dd>
    <dt>한 주소 제외</dt>
    <dd>브롱크스 북부의 한 주소가 1년에 4만7천 건(대부분 이웃 소음)을 신고했습니다(8장). 이 주소를 넣으면 브롱크스·겨울·앱 비중이 크게 부풀기 때문에, <b>소음·날씨·경찰·신고 경로</b>를 다룬 질문에서는 이 주소를 빼고 셌습니다. 원자료의 주소 칸은 띄어쓰기가 고르지 않아, 띄어쓰기를 정리한 주소로 걸러 냈습니다.</dd>
    <dt>경찰 처리 결과</dt>
    <dd>경찰이 남긴 결과 문장을 읽어 '흔적 없음', '현장 정리', '소환장(딱지)', '체포', '들어가지 못함' 등으로 나눴습니다. 가장 흔한 문장이 "소환장을 발부할 필요 없이 정리했다"여서 '소환장'이라는 낱말만 찾으면 딱지 비율이 세 배로 부풀므로, <b>"소환장을 발부했다"는 구절</b>이 있을 때만 딱지로 셌습니다. 다른 기관의 '가 보니 없었다', '이미 접수됨' 같은 답도 같은 방식으로 문장을 찾아 셌습니다.</dd>
    <dt>처리 시간</dt>
    <dd>종료된 민원 가운데 걸린 시간이 0보다 크고, 상태와 날짜가 서로 맞으며, <b>실제 처리 시각이 없는 기록(자정에 한꺼번에 닫힌 약 11.9만 건)이 아닌 것</b>만 썼습니다. 평균 대신 중앙값(한가운데 값)을 써서 아주 오래 걸린 몇 건에 끌려가지 않게 했습니다.</dd>
    <dt>그 밖의 규칙</dt>
    <dd>1년에 0건인 민원 종류는 종류 수에 넣지 않았습니다(그래서 194종). 비율은 소수 첫째 자리에서 반올림했고, 배수는 10배 이상이면 정수, 10배 미만이면 소수 한 자리로 적었습니다. 개인 거주지일 수 있는 주소는 동네만 적었습니다.</dd>
  </dl>
  <div class="src">자료: NYC Open Data, 311 Service Requests from 2010 to Present (전처리본) · Open-Meteo Historical Weather API · U.S. Census 2020.<br>
  모든 숫자와 그림은 같은 계산 과정으로 다시 만들 수 있습니다.</div>
</section>

</body>
</html>
""")
(ROOT / "reports" / "book" / "nyc311_hundred_questions.html").write_text("".join(out), encoding="utf-8")
print("ok · 질문 쪽", n_pages_q, "· 예상 총 쪽수", 2 + 12 + n_pages_q + 2)
