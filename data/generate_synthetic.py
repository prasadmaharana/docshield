"""
Synthetic PII document generator for DocShield training data.
Outputs YOLO-annotated PNG images of fake financial documents.

Usage:
    python data/generate_synthetic.py --count 100             # quick test
    python data/generate_synthetic.py --count 5000            # full dataset
    python data/generate_synthetic.py --count 5000 --output data/synthetic
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pypdfium2

# ── constants ──────────────────────────────────────────────────────────────────

CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]
CID = {c: i for i, c in enumerate(CLASSES)}

DPI = 150
SCALE = DPI / 72.0          # ReportLab points → screen pixels
PAGE_W, PAGE_H = A4         # 595.27 × 841.89 pts

fake = Faker()
Faker.seed(None)             # non-deterministic runs


# ── coordinate helpers ─────────────────────────────────────────────────────────

def rl_to_yolo(xl, yb, xr, yt, img_w, img_h):
    """
    Convert a ReportLab bounding rect to YOLO normalised format.

    ReportLab origin is bottom-left; image origin is top-left.
    xl / xr  : left and right x in pts
    yb / yt  : bottom and top y in pts (yb < yt in RL coords)
    Returns  : (cx, cy, w, h) all in [0, 1]
    """
    px_l = xl * SCALE / img_w
    px_r = xr * SCALE / img_w
    py_t = (PAGE_H - yt) * SCALE / img_h   # yt high in RL → low in image
    py_b = (PAGE_H - yb) * SCALE / img_h   # yb low  in RL → high in image
    cx = (px_l + px_r) / 2
    cy = (py_t + py_b) / 2
    w  = px_r - px_l
    h  = py_b - py_t
    return cx, cy, w, h


def text_rect(c, text, x, y, font, size):
    """Return (xl, yb, xr, yt) in RL pts for a string drawn at (x, y baseline)."""
    tw = c.stringWidth(text, font, size)
    return x, y - size * 0.20, x + tw, y + size * 0.75


def draw_value(c, text, x, y, font="Helvetica", size=11):
    """Draw text on canvas and return its raw RL rect."""
    c.setFont(font, size)
    c.drawString(x, y, text)
    return text_rect(c, text, x, y, font, size)


def draw_field(c, label, value, x, y, font="Helvetica", size=11, gap=130):
    """
    Draw 'Label: value' on one line.
    Returns the raw RL rect covering the *value* only (not the label).
    """
    c.setFont(font + "-Bold", size)
    c.drawString(x, y, label + ":")
    vx = x + gap
    c.setFont(font, size)
    c.drawString(vx, y, value)
    return text_rect(c, value, vx, y, font, size)


# ── fake PII helpers ───────────────────────────────────────────────────────────

def fake_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def fake_account(length=None):
    n = length or random.randint(10, 16)
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def fake_address():
    return fake.address().replace("\n", ", ")[:60]


def fake_pii():
    return {
        "name":    fake.name(),
        "ssn":     fake_ssn(),
        "account": fake_account(),
        "address": fake_address(),
        "phone":   fake.phone_number()[:22],
        "email":   fake.email(),
        "dob":     fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y"),
    }


# ── shared layout helpers ──────────────────────────────────────────────────────

def ruled_divider(c, y, margin):
    c.setStrokeColorRGB(0.15, 0.25, 0.60)
    c.setLineWidth(1.5)
    c.line(margin, y, PAGE_W - margin, y)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    return y - 18


def section_banner(c, y, margin, title):
    c.setFillColorRGB(0.12, 0.18, 0.55)
    c.rect(margin, y - 2, PAGE_W - 2 * margin, 15, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 4, y, title)
    c.setFillColorRGB(0, 0, 0)
    return y - 20


# ── document generators ────────────────────────────────────────────────────────
# Each returns (pdf_bytes, raw_annotations).
# raw_annotations : list of (class_id, xl_pts, yb_pts, xr_pts, yt_pts)

def gen_bank_statement():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    anns = []
    p = fake_pii()
    margin, y = 50, PAGE_H - 55

    # ── header ──────────────────────────────────────────────────────────────
    bank = random.choice([
        "First National Bank", "Pacific Trust Bank",
        "Apex Financial Group", "Heritage Bank & Trust",
    ])
    c.setFillColorRGB(0.10, 0.20, 0.60)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, bank)
    c.setFillColorRGB(0, 0, 0)
    y -= 14
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.45, 0.45, 0.45)
    c.drawString(margin, y, "Monthly Account Statement")
    c.setFillColorRGB(0, 0, 0)
    y -= 6
    y = ruled_divider(c, y, margin)

    # ── account holder block ─────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "ACCOUNT HOLDER")
    y -= 15

    gap, rh = 115, 16
    for label, key, cls in [
        ("Name",          "name",    "PERSON_NAME"),
        ("Account No.",   "account", "ACCOUNT_NUMBER"),
        ("Address",       "address", "ADDRESS"),
        ("Phone",         "phone",   "PHONE"),
        ("Email",         "email",   "EMAIL"),
        ("Date of Birth", "dob",     "DATE_OF_BIRTH"),
    ]:
        rect = draw_field(c, label, p[key], margin, y, gap=gap)
        anns.append((CID[cls], *rect))
        y -= rh

    y -= 8
    y = ruled_divider(c, y, margin)

    # ── statement period ─────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "STATEMENT PERIOD")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(
        margin, y,
        f"{fake.date_this_year().strftime('%B %d, %Y')}  –  {fake.date_this_year().strftime('%B %d, %Y')}",
    )
    y -= 18
    y = ruled_divider(c, y, margin)

    # ── transactions table ───────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "TRANSACTIONS")
    y -= 6
    c.line(margin, y, PAGE_W - margin, y)
    y -= 13

    cols = [margin, margin + 65, margin + 265, margin + 330, margin + 400]
    c.setFont("Helvetica-Bold", 8)
    for header, cx_ in zip(["Date", "Description", "Debit ($)", "Credit ($)", "Balance ($)"], cols):
        c.drawString(cx_, y, header)
    y -= 4
    c.line(margin, y, PAGE_W - margin, y)
    y -= 11

    balance = round(random.uniform(1_000, 80_000), 2)
    c.setFont("Helvetica", 8)
    for _ in range(random.randint(9, 15)):
        if y < 100:
            break
        desc = random.choice([
            "ATM Withdrawal", "Direct Deposit", "Online Transfer",
            f"Check #{random.randint(1000,9999)}", f"POS - {fake.company()[:18]}",
            "Wire Transfer", "ACH Debit", "Mortgage Payment", "Utility Bill",
        ])
        amount = round(random.uniform(15, 4_500), 2)
        date_s = fake.date_this_year().strftime("%m/%d/%Y")
        is_debit = random.random() < 0.45
        balance = round(balance - amount if is_debit else balance + amount, 2)
        c.drawString(cols[0], y, date_s)
        c.drawString(cols[1], y, desc[:28])
        c.drawString(cols[2 if is_debit else 3], y, f"{amount:,.2f}")
        c.drawString(cols[4], y, f"{balance:,.2f}")
        y -= 11

    # ── footer ───────────────────────────────────────────────────────────────
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(margin, 60, "Confidential — For account holder use only. Unauthorised distribution is prohibited.")
    c.setFillColorRGB(0, 0, 0)

    c.save()
    return buf.getvalue(), anns


def gen_kyc_form():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    anns = []
    p = fake_pii()
    margin, y = 50, PAGE_H - 48

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAGE_W / 2, y, "KNOW YOUR CUSTOMER (KYC) FORM")
    y -= 12
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(PAGE_W / 2, y, "Please complete all fields. Print clearly in BLOCK LETTERS.")
    c.setFillColorRGB(0, 0, 0)
    y -= 8
    y = ruled_divider(c, y, margin)

    gap, rh = 135, 18

    # ── personal information ─────────────────────────────────────────────────
    y = section_banner(c, y, margin, "SECTION 1 — PERSONAL INFORMATION")
    for label, key, cls in [
        ("Full Name",           "name",    "PERSON_NAME"),
        ("Social Security No.", "ssn",     "SSN"),
        ("Date of Birth",       "dob",     "DATE_OF_BIRTH"),
        ("Home Address",        "address", "ADDRESS"),
        ("Phone Number",        "phone",   "PHONE"),
        ("Email Address",       "email",   "EMAIL"),
    ]:
        rect = draw_field(c, label, p[key], margin, y, gap=gap)
        anns.append((CID[cls], *rect))
        y -= rh

    y -= 5
    y = section_banner(c, y, margin, "SECTION 2 — ACCOUNT INFORMATION")

    rect = draw_field(c, "Account Number", p["account"], margin, y, gap=gap)
    anns.append((CID["ACCOUNT_NUMBER"], *rect))
    y -= rh

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Account Type:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + gap, y, random.choice(["Checking", "Savings", "Investment", "Business"]))
    y -= rh + 5

    y = section_banner(c, y, margin, "SECTION 3 — IDENTITY VERIFICATION")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "ID Type:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + gap, y, random.choice(["Passport", "Driver License", "State ID", "Military ID"]))
    y -= rh

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "ID Number:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + gap, y, fake.bothify("??######??").upper())
    y -= rh + 15

    # ── declaration ──────────────────────────────────────────────────────────
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    decl = (
        "I hereby certify that the information provided is true and accurate. "
        "I consent to the verification and processing of my personal data in "
        "accordance with applicable regulations."
    )
    words, line = decl.split(), ""
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, "Helvetica", 8) < PAGE_W - 2 * margin:
            line = test
        else:
            c.drawString(margin, y, line)
            y -= 11
            line = word
    if line:
        c.drawString(margin, y, line)
        y -= 18
    c.setFillColorRGB(0, 0, 0)

    # ── signature block ───────────────────────────────────────────────────────
    sig_y = y - 28
    c.setLineWidth(0.5)
    c.line(margin, sig_y, margin + 190, sig_y)
    c.setFont("Helvetica-Oblique", 13)
    sig_text = p["name"]
    c.drawString(margin + 4, sig_y + 6, sig_text)
    sig_tw = c.stringWidth(sig_text, "Helvetica-Oblique", 13)
    anns.append((CID["SIGNATURE"], margin, sig_y - 4, margin + sig_tw + 10, sig_y + 16))
    c.setFont("Helvetica", 8)
    c.drawString(margin, sig_y - 11, "Applicant Signature")

    date_x = PAGE_W - margin - 130
    c.setFont("Helvetica", 11)
    c.drawString(date_x, sig_y + 6, fake.date_this_year().strftime("%m/%d/%Y"))
    c.line(date_x - 5, sig_y, date_x + 125, sig_y)
    c.setFont("Helvetica", 8)
    c.drawString(date_x, sig_y - 11, "Date")

    c.save()
    return buf.getvalue(), anns


def gen_loan_application():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    anns = []
    p = fake_pii()
    margin, y = 50, PAGE_H - 48

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(PAGE_W / 2, y, "PERSONAL LOAN APPLICATION")
    y -= 12
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(
        PAGE_W / 2, y,
        f"Ref: {fake.bothify('LN-######').upper()}  ·  {fake.date_this_year().strftime('%B %d, %Y')}",
    )
    c.setFillColorRGB(0, 0, 0)
    y -= 8
    y = ruled_divider(c, y, margin)

    gap, rh = 145, 17

    # ── borrower info ─────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "BORROWER INFORMATION")
    y -= 14

    for label, key, cls in [
        ("Borrower Name",   "name",    "PERSON_NAME"),
        ("SSN",             "ssn",     "SSN"),
        ("Date of Birth",   "dob",     "DATE_OF_BIRTH"),
        ("Current Address", "address", "ADDRESS"),
        ("Phone",           "phone",   "PHONE"),
        ("Email",           "email",   "EMAIL"),
    ]:
        rect = draw_field(c, label, p[key], margin, y, gap=gap)
        anns.append((CID[cls], *rect))
        y -= rh

    y -= 8
    c.setLineWidth(0.4)
    c.line(margin, y, PAGE_W - margin, y)
    y -= 12

    # ── banking info ─────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "BANKING INFORMATION")
    y -= 14

    rect = draw_field(c, "Account Number", p["account"], margin, y, gap=gap)
    anns.append((CID["ACCOUNT_NUMBER"], *rect))
    y -= rh

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Bank Name:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + gap, y, random.choice(["Chase Bank", "Wells Fargo", "Bank of America", "Citibank"]))
    y -= rh

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Routing Number:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + gap, y, "".join(str(random.randint(0, 9)) for _ in range(9)))
    y -= rh + 8

    c.setLineWidth(0.4)
    c.line(margin, y, PAGE_W - margin, y)
    y -= 12

    # ── loan details ──────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "LOAN DETAILS")
    y -= 14

    for label, value in [
        ("Requested Amount", f"${random.choice([5000, 10000, 25000, 50000, 100000]):,}"),
        ("Loan Term",        f"{random.choice([12, 24, 36, 48, 60])} months"),
        ("Purpose",          random.choice(["Home Improvement", "Debt Consolidation", "Medical", "Education", "Business"])),
        ("Employer",         fake.company()[:40]),
        ("Annual Income",    f"${random.randint(35_000, 200_000):,}"),
    ]:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, label + ":")
        c.setFont("Helvetica", 11)
        c.drawString(margin + gap, y, value)
        y -= rh

    y -= 20

    # ── signature ─────────────────────────────────────────────────────────────
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(margin, y, "By signing, I certify all information is accurate and authorise credit verification.")
    c.setFillColorRGB(0, 0, 0)
    y -= 35

    c.setLineWidth(0.5)
    c.line(margin, y, margin + 205, y)
    c.setFont("Helvetica-Oblique", 13)
    sig_text = p["name"]
    c.drawString(margin + 4, y + 6, sig_text)
    sig_tw = c.stringWidth(sig_text, "Helvetica-Oblique", 13)
    anns.append((CID["SIGNATURE"], margin, y - 4, margin + sig_tw + 10, y + 16))
    c.setFont("Helvetica", 8)
    c.drawString(margin, y - 11, "Borrower Signature & Date")

    c.save()
    return buf.getvalue(), anns


def gen_contract():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    anns = []
    p1, p2 = fake_pii(), fake_pii()
    margin, y = 50, PAGE_H - 48

    contract_type = random.choice([
        "SERVICE AGREEMENT", "NON-DISCLOSURE AGREEMENT", "CONSULTING AGREEMENT",
    ])
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(PAGE_W / 2, y, contract_type)
    y -= 14
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(PAGE_W / 2, y, f"Effective Date: {fake.date_this_year().strftime('%B %d, %Y')}")
    c.setFillColorRGB(0, 0, 0)
    y -= 8
    y = ruled_divider(c, y, margin)

    gap, rh = 110, 16
    for party_label, p in [("PARTY A — SERVICE PROVIDER", p1), ("PARTY B — CLIENT", p2)]:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(margin, y, party_label)
        y -= 14
        for label, key, cls in [
            ("Name",    "name",    "PERSON_NAME"),
            ("Address", "address", "ADDRESS"),
            ("Phone",   "phone",   "PHONE"),
            ("Email",   "email",   "EMAIL"),
        ]:
            rect = draw_field(c, label, p[key], margin, y, gap=gap)
            anns.append((CID[cls], *rect))
            y -= rh
        y -= 10
        c.setLineWidth(0.4)
        c.line(margin, y, PAGE_W - margin, y)
        y -= 12

    # ── terms ────────────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, y, "TERMS AND CONDITIONS")
    y -= 14
    c.setFont("Helvetica", 9)
    for term in [
        "1. Both parties agree to maintain strict confidentiality of all shared information.",
        "2. Services shall be rendered as described in Schedule A (attached).",
        "3. Payment terms: Net 30 days from date of invoice.",
        "4. This agreement is governed by the laws of the applicable jurisdiction.",
        "5. Either party may terminate with 30 days written notice.",
        "6. Intellectual property created under this agreement remains with the service provider.",
        "7. Disputes shall be resolved through binding arbitration.",
    ]:
        if y < 175:
            break
        c.drawString(margin, y, term)
        y -= 13

    y -= 22

    # ── two signature blocks ───────────────────────────────────────────────────
    sig_font = 13
    for sig_x, p, label_text in [
        (margin,           p1, "Party A Signature"),
        (PAGE_W / 2 + 15, p2, "Party B Signature"),
    ]:
        c.setLineWidth(0.5)
        c.line(sig_x, y, sig_x + 188, y)
        c.setFont("Helvetica-Oblique", sig_font)
        sig_text = p["name"]
        c.drawString(sig_x + 5, y + 6, sig_text)
        sig_tw = c.stringWidth(sig_text, "Helvetica-Oblique", sig_font)
        anns.append((CID["SIGNATURE"], sig_x, y - 4, sig_x + sig_tw + 10, y + sig_font + 5))
        c.setFont("Helvetica", 8)
        c.drawString(sig_x, y - 11, label_text)

    c.save()
    return buf.getvalue(), anns


def gen_cheque():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    anns = []
    p = fake_pii()
    payee = fake.name()
    margin = 50

    # Cheque occupies upper ~30% of the page
    ck_l, ck_r = margin, PAGE_W - margin
    ck_t, ck_b = PAGE_H - 90, PAGE_H - 340

    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.8)
    c.rect(ck_l, ck_b, ck_r - ck_l, ck_t - ck_b)

    y = ck_t - 18

    # ── payer name + address (top-left) ──────────────────────────────────────
    c.setFont("Helvetica-Bold", 11)
    c.drawString(ck_l + 10, y, p["name"])
    name_tw = c.stringWidth(p["name"], "Helvetica-Bold", 11)
    anns.append((CID["PERSON_NAME"], ck_l + 10, y - 2, ck_l + 10 + name_tw, y + 9))
    y -= 13

    addr_text = p["address"][:55]
    c.setFont("Helvetica", 9)
    c.drawString(ck_l + 10, y, addr_text)
    addr_tw = c.stringWidth(addr_text, "Helvetica", 9)
    anns.append((CID["ADDRESS"], ck_l + 10, y - 2, ck_l + 10 + addr_tw, y + 7))
    y -= 12

    c.setFont("Helvetica", 8)
    c.drawString(ck_l + 10, y, p["phone"])
    y -= 30

    # ── cheque number + date (top-right) ─────────────────────────────────────
    ck_num = str(random.randint(1001, 9999))
    c.setFont("Courier", 10)
    c.drawRightString(ck_r - 10, ck_t - 18, f"No. {ck_num}")
    c.setFont("Helvetica", 10)
    c.drawRightString(ck_r - 10, y + 14, f"Date: {fake.date_this_year().strftime('%m/%d/%Y')}")

    # ── pay-to line ───────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 10)
    c.drawString(ck_l + 10, y, "PAY TO THE ORDER OF")
    pay_x = ck_l + 148
    c.setFont("Helvetica", 11)
    c.drawString(pay_x, y, payee)
    pay_tw = c.stringWidth(payee, "Helvetica", 11)
    anns.append((CID["PERSON_NAME"], pay_x, y - 2, pay_x + pay_tw, y + 9))

    # Amount box
    amount = round(random.uniform(50, 9_999), 2)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.rect(ck_r - 100, y - 3, 88, 15)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(ck_r - 56, y, f"$ {amount:,.2f}")
    y -= 20

    # Legal amount
    c.setLineWidth(0.3)
    c.line(ck_l + 10, y, ck_r - 105, y)
    y -= 12
    dollars, cents = int(amount), int(round((amount - int(amount)) * 100))
    c.setFont("Helvetica", 9)
    c.drawString(ck_l + 10, y, f"*** {dollars:,} dollars and {cents:02d}/100 CENTS ***")
    y -= 20

    # ── bank name ─────────────────────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 10)
    c.drawString(ck_l + 10, y, random.choice(["First National Bank", "Pacific Trust Bank", "Apex Bank"]))

    # ── MICR account number (bottom of cheque) ───────────────────────────────
    micr_y = ck_b + 18
    acct_micr = p["account"][:16]
    c.setFont("Courier-Bold", 10)
    c.drawString(ck_l + 10, micr_y, acct_micr)
    acct_tw = c.stringWidth(acct_micr, "Courier-Bold", 10)
    anns.append((CID["ACCOUNT_NUMBER"], ck_l + 10, micr_y - 2, ck_l + 10 + acct_tw, micr_y + 8))

    routing = "".join(str(random.randint(0, 9)) for _ in range(9))
    c.drawString(ck_l + 10 + acct_tw + 28, micr_y, f"{routing}")

    # ── signature (bottom-right of cheque) ───────────────────────────────────
    sig_x = ck_r - 200
    sig_y = ck_b + 42
    c.setLineWidth(0.5)
    c.line(sig_x, sig_y, ck_r - 10, sig_y)
    c.setFont("Helvetica-Oblique", 12)
    sig_text = p["name"]
    c.drawString(sig_x + 4, sig_y + 5, sig_text)
    sig_tw = c.stringWidth(sig_text, "Helvetica-Oblique", 12)
    anns.append((CID["SIGNATURE"], sig_x, sig_y - 3, sig_x + sig_tw + 8, sig_y + 14))
    c.setFont("Helvetica", 7)
    c.drawString(sig_x + 4, sig_y - 9, "Authorized Signature")

    # ── memo (below cheque box) ───────────────────────────────────────────────
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.45, 0.45, 0.45)
    c.drawString(margin, ck_b - 28, "MEMO: " + random.choice([
        "Rent Payment", f"Invoice #{random.randint(100, 9999)}",
        "Services Rendered", "Security Deposit", "Loan Repayment",
    ]))
    c.setFillColorRGB(0, 0, 0)

    c.save()
    return buf.getvalue(), anns


# ── rendering & augmentation ──────────────────────────────────────────────────

GENERATORS = [gen_bank_statement, gen_kyc_form, gen_loan_application, gen_contract, gen_cheque]


def pdf_to_pil(pdf_bytes: bytes) -> Image.Image:
    """Render the first page of a PDF to an RGB PIL image at DPI resolution."""
    doc = pypdfium2.PdfDocument(pdf_bytes)
    page = doc.get_page(0)
    bitmap = page.render(scale=SCALE)
    img = bitmap.to_pil().convert("RGB")
    page.close()
    doc.close()
    return img


def apply_scan_effects(img: Image.Image) -> Image.Image:
    """
    Simulate scanned-document artefacts:
      • ±2° random rotation     (physical mis-feed)
      • ±18% brightness jitter  (scanner exposure)
      • ±12% contrast jitter    (scanner gain)
      • Gaussian pixel noise    (sensor noise)
      • 30% chance of soft blur (scan softness / slight focus error)
    """
    angle = random.uniform(-2.0, 2.0)
    img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.82, 1.18))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.88, 1.12))

    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0.0, random.uniform(1.5, 6.0), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))

    return img


# ── dataset writer ────────────────────────────────────────────────────────────

def generate_dataset(count: int, output_dir: str) -> None:
    out = Path(output_dir)
    imgs_dir   = out / "images"
    labels_dir = out / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {count} documents -> {out}/")

    for i in range(count):
        gen_fn = random.choice(GENERATORS)
        pdf_bytes, raw_anns = gen_fn()

        img = pdf_to_pil(pdf_bytes)
        img_w, img_h = img.size

        # Normalise bboxes against the actual rendered image dimensions
        yolo_anns = []
        for class_id, xl, yb, xr, yt in raw_anns:
            cx, cy, w, h = rl_to_yolo(xl, yb, xr, yt, img_w, img_h)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.001, min(1.0, w))
            h  = max(0.001, min(1.0, h))
            yolo_anns.append((class_id, cx, cy, w, h))

        img = apply_scan_effects(img)

        name = f"doc_{i:05d}"
        img.save(imgs_dir / f"{name}.png")
        with open(labels_dir / f"{name}.txt", "w") as f:
            for class_id, cx, cy, w, h in yolo_anns:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if (i + 1) % 50 == 0 or i == count - 1:
            print(f"  [{i + 1:5d}/{count}]  {gen_fn.__name__:<25}  {img_w}×{img_h} px", flush=True)

    print(f"\nDataset complete: {count} images + labels in {out}/")
    print(f"YOLO class map: {', '.join(f'{i}={c}' for i, c in enumerate(CLASSES))}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocShield — synthetic PII document generator")
    parser.add_argument("--count",  type=int, default=5000,            help="Number of images to generate")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()
    generate_dataset(args.count, args.output)


if __name__ == "__main__":
    main()
