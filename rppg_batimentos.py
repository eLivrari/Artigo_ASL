"""
Estimativa de frequencia cardiaca (BPM) a partir de video facial - rPPG
(fotopletismografia remota).

Pipeline completo (FFT, STFT e CWT):
  1. Deteccao do rosto (Haar cascade do OpenCV) e definicao de ROIs de pele:
     testa e bochechas (evita olhos, boca e barba).
  2. Media RGB das ROIs em cada quadro -> sinal bruto RGB(t).
  3. Extracao do sinal de pulso pelos metodos:
       - GREEN  (Verkruysse et al., 2008)
       - CHROM  (de Haan & Jeanne, 2013)
       - POS    (Wang et al., 2017)  <- metodo principal
  4. Filtro passa-banda Butterworth 0,7-3,5 Hz (42-210 BPM) em fase zero.
  5. Caracterizacao de nao estacionariedade (media e variancia deslizantes).
  6. Estimacao comparativa de BPM via:
       - FFT global (espectro completo / Welch)
       - STFT (espectrograma, janela Hann 10 s, passo 1 s)
       - CWT (escalograma, wavelet de Morlet complexa cmor2.0-0.955, 128 escalas)

Saidas (pasta resultados/):
  <video>_roi.jpg               frame de exemplo com deteccao facial e ROIs
  <video>_bruto_vs_filtrado.png sinal bruto vs. filtrado + media/variancia deslizantes
  <video>_sinal.png             sinal rPPG filtrado (POS) + espectro global
  <video>_tempo_frequencia.png  espectrograma (STFT) e escalograma (CWT)
  <video>_bpm_tempo.png         comparacao temporal STFT vs. CWT e entre metodos
  <video>_bpm.csv               series temporais de BPM (STFT e CWT)
  resumo.csv                    tabela consolidada (FFT, STFT e CWT) para o artigo
"""
import argparse
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal

F_MIN, F_MAX = 0.7, 3.5          # Hz  (42 - 210 BPM)
JANELA_S, PASSO_S = 10.0, 1.0    # janela deslizante STFT
DESCARTE_INICIO_S = 1.0          # ignora o 1o segundo (ajuste de camera)
DESCARTE_BORDA_S = 5.0           # descarta 5 s nas extremidades para series temporais

REFERENCIAS_PADRAO = {
    "SemMovimento": (65, 75),
    "ComMovimento": (90, 100),
}


# ----------------------------------------------------------------------------
# 1-2. Leitura do video e extracao do sinal RGB
# ----------------------------------------------------------------------------
def rois_do_rosto(x, y, w, h):
    """ROIs relativas a caixa do rosto: testa e bochechas."""
    testa = (x + int(0.30 * w), y + int(0.08 * h), int(0.40 * w), int(0.15 * h))
    boch_e = (x + int(0.15 * w), y + int(0.50 * h), int(0.20 * w), int(0.15 * h))
    boch_d = (x + int(0.65 * w), y + int(0.50 * h), int(0.20 * w), int(0.15 * h))
    return [testa, boch_e, boch_d]


def mascara_pele(roi_bgr):
    """Mascara simples de pele em YCrCb para descartar pixels nao-pele."""
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return m > 0


def extrair_rgb(caminho, salvar_preview=None):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    rgb, caixa_suave, n = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        esc = 0.5
        faces = detector.detectMultiScale(
            cv2.resize(cinza, None, fx=esc, fy=esc), 1.1, 5, minSize=(60, 60)
        )
        if len(faces):
            f = max(faces, key=lambda r: r[2] * r[3]) / esc
            caixa_suave = f if caixa_suave is None else 0.85 * caixa_suave + 0.15 * f
        if caixa_suave is None:
            rgb.append([np.nan] * 3)
            continue

        x, y, w, h = caixa_suave.astype(int)
        pixels = []
        for (rx, ry, rw, rh) in rois_do_rosto(x, y, w, h):
            roi = frame[max(ry, 0):ry + rh, max(rx, 0):rx + rw]
            if roi.size == 0:
                continue
            m = mascara_pele(roi)
            pixels.append(roi[m] if m.sum() > 50 else roi.reshape(-1, 3))
        px = np.concatenate(pixels).astype(np.float64)
        b, g, r = px.mean(axis=0)
        rgb.append([r, g, b])

        if salvar_preview and n == 60:
            prev = frame.copy()
            cv2.rectangle(prev, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (rx, ry, rw, rh) in rois_do_rosto(x, y, w, h):
                cv2.rectangle(prev, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.imwrite(salvar_preview, prev)
    cap.release()

    rgb = np.array(rgb)
    for c in range(3):
        col = rgb[:, c]
        nan = np.isnan(col)
        if nan.all():
            raise RuntimeError("Nenhum rosto detectado no video.")
        col[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), col[~nan])
    return rgb, fps


# ----------------------------------------------------------------------------
# 3. Metodos de extracao do pulso
# ----------------------------------------------------------------------------
def metodo_green(rgb, fps):
    return rgb[:, 1] - rgb[:, 1].mean()


def metodo_chrom(rgb, fps):
    win = int(1.6 * fps)
    N = len(rgb)
    h = np.zeros(N)
    b_, a_ = signal.butter(3, [F_MIN, F_MAX], btype="band", fs=fps)
    for s in range(0, N - win, win // 2):
        C = rgb[s:s + win]
        Cn = C / C.mean(axis=0)
        X = 3 * Cn[:, 0] - 2 * Cn[:, 1]
        Y = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
        Xf = signal.filtfilt(b_, a_, X, padlen=min(len(X) - 1, 9))
        Yf = signal.filtfilt(b_, a_, Y, padlen=min(len(Y) - 1, 9))
        alpha = Xf.std() / (Yf.std() + 1e-9)
        S = (Xf - alpha * Yf) * signal.windows.hann(win)
        h[s:s + win] += S
    return h


def metodo_pos(rgb, fps):
    win = int(1.6 * fps)
    N = len(rgb)
    H = np.zeros(N)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for m in range(win, N + 1):
        s = m - win
        C = rgb[s:m].T
        Cn = C / C.mean(axis=1, keepdims=True)
        S = P @ Cn
        alpha = S[0].std() / (S[1].std() + 1e-9)
        h = S[0] + alpha * S[1]
        H[s:m] += h - h.mean()
    return H


METODOS = {"POS": metodo_pos, "CHROM": metodo_chrom, "GREEN": metodo_green}


# ----------------------------------------------------------------------------
# 4-6. Filtragem, Nao Estacionariedade e Estimadores (FFT, STFT, CWT)
# ----------------------------------------------------------------------------
def filtrar(x, fps):
    b, a = signal.butter(3, [F_MIN, F_MAX], btype="band", fs=fps)
    x = signal.detrend(x)
    return signal.filtfilt(b, a, x)


def estatisticas_deslizantes(x, fps, win_s=3.0, passo_s=0.5):
    """Media e variancia locais em janelas deslizantes (Eqs. 9 e 10)."""
    L = int(win_s * fps)
    D = max(1, int(passo_s * fps))
    t_w, mu_w, var_w = [], [], []
    for s in range(0, len(x) - L + 1, D):
        seg = x[s:s + L]
        t_w.append((s + L / 2) / fps)
        mu_w.append(np.mean(seg))
        var_w.append(np.var(seg, ddof=1))
    return np.array(t_w), np.array(mu_w), np.array(var_w)


def bpm_fft_global(x, fps):
    """Estimativa global via FFT com zero-padding (Eqs. 5-8)."""
    nfft = max(4096, int(2 ** np.ceil(np.log2(len(x)))) * 4)
    X = np.abs(np.fft.rfft(x, n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fps)
    banda = (f >= F_MIN) & (f <= F_MAX)
    f_b, p_b = f[banda], X[banda]
    return 60.0 * f_b[np.argmax(p_b)], f_b * 60.0, p_b


def bpm_stft(x, fps):
    """Estimativa tempo-frequencia via STFT (Eqs. 11-12)."""
    nw = int(JANELA_S * fps)
    r = int(PASSO_S * fps)
    f, t, Sxx = signal.spectrogram(
        x, fs=fps, window="hann", nperseg=nw, noverlap=nw - r, nfft=4096, mode="psd"
    )
    banda = (f >= F_MIN) & (f <= F_MAX)
    f_b, S_b = f[banda], Sxx[banda, :]
    idx_max = np.argmax(S_b, axis=0)
    bpm_t = 60.0 * f_b[idx_max]
    return t, f_b * 60.0, S_b, bpm_t


def bpm_cwt(x, fps, n_escalas=128):
    """Estimativa multiescala via CWT de Morlet complexa (Eqs. 13-16)."""
    wavelet = "cmor2.0-0.955"  # omega_0 = 6 -> fc = 6 / (2*pi) ~= 0.955
    fc = pywt.central_frequency(wavelet)
    freqs = np.geomspace(F_MIN, F_MAX, n_escalas)
    escalas = (fc * fps) / freqs
    coef, _ = pywt.cwt(x, escalas, wavelet, sampling_period=1.0 / fps)
    escalograma = np.abs(coef) ** 2

    # Subamostra a cada 1 segundo (mesma grade temporal de 1 Hz para comparacao)
    passo = int(PASSO_S * fps)
    indices = np.arange(0, len(x), passo)
    t = indices / fps
    esc_sub = escalograma[:, indices]
    idx_max = np.argmax(esc_sub, axis=0)
    bpm_t = 60.0 * freqs[idx_max]
    return t, freqs * 60.0, escalograma, bpm_t


def aparar_bordas(t, bpm_t, duracao_s, borda_s=DESCARTE_BORDA_S):
    """Descarta bordas para evitar efeitos de extremidade (Secao III-E)."""
    mask = (t >= borda_s) & (t <= (duracao_s - borda_s))
    if mask.sum() == 0:
        return t, bpm_t
    return t[mask], bpm_t[mask]


# ----------------------------------------------------------------------------
def processar(caminho, pasta, referencia=None):
    nome = os.path.splitext(os.path.basename(caminho))[0]
    print(f"\n=== {nome} ===")
    rgb, fps = extrair_rgb(caminho, os.path.join(pasta, f"{nome}_roi.jpg"))
    rgb = rgb[int(DESCARTE_INICIO_S * fps):]
    duracao_s = len(rgb) / fps
    print(f"  FPS: {fps:.2f} | quadros: {len(rgb)} | duracao: {duracao_s:.1f} s")

    tt = np.arange(len(rgb)) / fps + DESCARTE_INICIO_S

    # 1. Grafico: Dado Bruto vs. Filtrado + Nao Estacionariedade (Eqs. 9 e 10)
    sinal_pos_bruto = metodo_pos(rgb, fps)
    sinal_pos_filt = filtrar(sinal_pos_bruto, fps)
    t_w, mu_w, var_w = estatisticas_deslizantes(rgb[:, 1], fps, win_s=3.0, passo_s=0.5)

    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=False)
    ax[0].plot(tt, rgb[:, 1], color="tab:green", lw=1)
    ax[0].set(title=f"{nome} - Canal Verde Bruto (com nivel DC e deriva de iluminacao)",
              ylabel="Intensidade (0-255)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(tt, sinal_pos_filt, color="tab:blue", lw=1)
    ax[1].set(title=f"{nome} - Sinal POS apos Detrend e Filtro Butterworth (0,7-3,5 Hz)",
              ylabel="Amplitude norm.")
    ax[1].grid(alpha=0.3)

    ax2_twin = ax[2].twinx()
    l1 = ax[2].plot(t_w + DESCARTE_INICIO_S, mu_w, color="tab:orange", lw=1.5, label="Media local $\\mu_w[m]$")
    l2 = ax2_twin.plot(t_w + DESCARTE_INICIO_S, var_w, color="tab:purple", ls="--", lw=1.5, label="Variancia local $\\sigma_w^2[m]$")
    ax[2].set(title=f"{nome} - Caracterizacao de Nao Estacionariedade (janela 3 s)",
              xlabel="Tempo (s)", ylabel="Media local (intensidade)")
    ax2_twin.set_ylabel("Variancia local")
    linhas_leg = l1 + l2
    ax[2].legend(linhas_leg, [l.get_label() for l in linhas_leg], loc="upper right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_bruto_vs_filtrado.png"), dpi=150)
    plt.close(fig)

    # 2. Executa os 3 metodos (POS, CHROM, GREEN) x 3 transformadas (FFT, STFT, CWT)
    resultados = {}
    for nm, fn in METODOS.items():
        sinal = filtrar(fn(rgb, fps), fps)

        # FFT global
        bpm_fft, fb_fft, pb_fft = bpm_fft_global(sinal, fps)

        # STFT
        t_stft, fb_stft, S_stft, bpm_stft_raw = bpm_stft(sinal, fps)
        t_stft_ap, bpm_stft_ap = aparar_bordas(t_stft, bpm_stft_raw, duracao_s)

        # CWT
        t_cwt, fb_cwt, E_cwt, bpm_cwt_raw = bpm_cwt(sinal, fps)
        t_cwt_ap, bpm_cwt_ap = aparar_bordas(t_cwt, bpm_cwt_raw, duracao_s)

        resultados[nm] = dict(
            sinal=sinal,
            bpm_fft=bpm_fft, fb_fft=fb_fft, pb_fft=pb_fft,
            t_stft=t_stft, fb_stft=fb_stft, S_stft=S_stft,
            t_stft_ap=t_stft_ap, bpm_stft=bpm_stft_ap,
            t_cwt=t_cwt, fb_cwt=fb_cwt, E_cwt=E_cwt,
            t_cwt_ap=t_cwt_ap, bpm_cwt=bpm_cwt_ap,
        )
        print(f"  {nm:6s} | FFT: {bpm_fft:5.1f} BPM | "
              f"STFT media: {bpm_stft_ap.mean():5.1f} (dp {bpm_stft_ap.std():4.1f}) | "
              f"CWT media: {bpm_cwt_ap.mean():5.1f} (dp {bpm_cwt_ap.std():4.1f})")

    # 3. Grafico: Sinal filtrado + Espectro FFT (POS)
    r = resultados["POS"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(tt, r["sinal"], lw=1)
    ax[0].set(title=f"{nome} - sinal rPPG (POS, filtrado 0,7-3,5 Hz)",
              xlabel="Tempo (s)", ylabel="Amplitude")
    ax[1].plot(r["fb_fft"], r["pb_fft"] / r["pb_fft"].max(), lw=1.5)
    ax[1].axvline(r["bpm_fft"], color="r", ls="--", label=f"pico FFT = {r['bpm_fft']:.1f} BPM")
    if referencia:
        ax[1].axvspan(*referencia, color="g", alpha=0.15,
                      label=f"referencia {referencia[0]}-{referencia[1]} BPM")
    ax[1].set(xlabel="Frequencia (BPM)", ylabel="Potencia (norm.)", xlim=(40, 200))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_sinal.png"), dpi=150)
    plt.close(fig)

    # 4. Grafico 2D: Espectrograma (STFT) vs. Escalograma (CWT Morlet) para POS
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    im0 = ax[0].pcolormesh(
        r["t_stft"] + DESCARTE_INICIO_S, r["fb_stft"],
        r["S_stft"] / (r["S_stft"].max() + 1e-12),
        shading="gouraud", cmap="viridis"
    )
    ax[0].plot(r["t_stft_ap"] + DESCARTE_INICIO_S, r["bpm_stft"], "w--", lw=1.5, label="Crista STFT")
    if referencia:
        ax[0].axhline(referencia[0], color="lime", ls=":", lw=1)
        ax[0].axhline(referencia[1], color="lime", ls=":", lw=1, label="Faixa ref.")
    ax[0].set(title=f"{nome} - Espectrograma STFT (POS)", xlabel="Tempo (s)", ylabel="Frequencia (BPM)", ylim=(42, 180))
    ax[0].legend(loc="upper right")
    fig.colorbar(im0, ax=ax[0], label="Potencia norm.")

    im1 = ax[1].pcolormesh(
        tt, r["fb_cwt"],
        r["E_cwt"] / (r["E_cwt"].max() + 1e-12),
        shading="gouraud", cmap="viridis"
    )
    ax[1].plot(r["t_cwt_ap"] + DESCARTE_INICIO_S, r["bpm_cwt"], "w--", lw=1.5, label="Crista CWT")
    if referencia:
        ax[1].axhline(referencia[0], color="lime", ls=":", lw=1)
        ax[1].axhline(referencia[1], color="lime", ls=":", lw=1, label="Faixa ref.")
    ax[1].set(title=f"{nome} - Escalograma CWT Morlet (POS)", xlabel="Tempo (s)", ylim=(42, 180))
    ax[1].legend(loc="upper right")
    fig.colorbar(im1, ax=ax[1], label="Energia norm.")

    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_tempo_frequencia.png"), dpi=150)
    plt.close(fig)

    # 5. Grafico Temporal: Comparacao STFT vs. CWT e entre Metodos (POS, CHROM, GREEN)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for nm, rr in resultados.items():
        ax[0].plot(rr["t_stft_ap"] + DESCARTE_INICIO_S, rr["bpm_stft"], marker="o", ms=3, label=f"{nm} (STFT)")
        ax[1].plot(rr["t_cwt_ap"] + DESCARTE_INICIO_S, rr["bpm_cwt"], marker="s", ms=3, label=f"{nm} (CWT)")

    for a_i, titulo in zip(ax, ["STFT (janela 10 s)", "CWT (Wavelet de Morlet)"]):
        if referencia:
            a_i.axhspan(*referencia, color="g", alpha=0.15, label="referencia (sensor)")
        a_i.set(title=f"{nome} - BPM ao longo do tempo via {titulo}", ylabel="BPM", ylim=(40, 160))
        a_i.legend(loc="upper right", ncol=2)
        a_i.grid(alpha=0.3)
    ax[1].set_xlabel("Tempo (s)")

    fig.tight_layout()
    fig.savefig(os.path.join(pasta, f"{nome}_bpm_tempo.png"), dpi=150)
    plt.close(fig)

    # 6. CSV das series temporais (STFT e CWT)
    with open(os.path.join(pasta, f"{nome}_bpm.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["tempo_s"]
            + [f"stft_{m}" for m in resultados]
            + [f"cwt_{m}" for m in resultados]
        )
        n_lin = min(len(r["t_stft_ap"]), len(r["t_cwt_ap"]))
        for i in range(n_lin):
            w.writerow(
                [f"{r['t_stft_ap'][i] + DESCARTE_INICIO_S:.1f}"]
                + [f"{resultados[m]['bpm_stft'][i]:.1f}" for m in resultados]
                + [f"{resultados[m]['bpm_cwt'][i]:.1f}" for m in resultados]
            )
    return nome, resultados


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*",
                    default=[os.path.join(base, "SemMovimento.mp4"),
                             os.path.join(base, "ComMovimento.mp4")])
    ap.add_argument("--ref", action="append", default=[],
                    help="referencia: NOME=min-max, ex. SemMovimento=65-75")
    ap.add_argument("--saida", default=os.path.join(base, "resultados"))
    a = ap.parse_args()

    refs = dict(REFERENCIAS_PADRAO)
    for item in a.ref:
        k, v = item.split("=")
        lo, hi = v.split("-")
        refs[os.path.splitext(k)[0]] = (float(lo), float(hi))

    os.makedirs(a.saida, exist_ok=True)
    linhas = []
    for v in a.videos:
        nome = os.path.splitext(os.path.basename(v))[0]
        ref = refs.get(nome)
        nome, res = processar(v, a.saida, ref)
        for m, r in res.items():
            # Linha 1: FFT global
            # Linha 2: STFT (media temporal)
            # Linha 3: CWT  (media temporal)
            entradas = [
                ("FFT",  r["bpm_fft"],          r["bpm_fft"],          r["bpm_fft"],          0.0),
                ("STFT", r["bpm_stft"].mean(),  r["bpm_stft"].min(),   r["bpm_stft"].max(),   r["bpm_stft"].std()),
                ("CWT",  r["bpm_cwt"].mean(),   r["bpm_cwt"].min(),    r["bpm_cwt"].max(),    r["bpm_cwt"].std()),
            ]
            for transf, bpm_est, bpm_min, bpm_max, dp in entradas:
                linha = dict(
                    video=nome,
                    metodo=m,
                    transformada=transf,
                    bpm_estimado=round(float(bpm_est), 1),
                    bpm_min=round(float(bpm_min), 1),
                    bpm_max=round(float(bpm_max), 1),
                    desvio_padrao=round(float(dp), 2),
                    ref_min=ref[0] if ref else "",
                    ref_max=ref[1] if ref else "",
                )
                if ref:
                    centro = (ref[0] + ref[1]) / 2.0
                    linha["erro_abs_vs_centro_ref"] = round(abs(float(bpm_est) - centro), 1)
                    linha["dentro_da_faixa"] = ref[0] <= float(bpm_est) <= ref[1]
                linhas.append(linha)

    campos = [
        "video", "metodo", "transformada", "bpm_estimado", "bpm_min", "bpm_max",
        "desvio_padrao", "ref_min", "ref_max", "erro_abs_vs_centro_ref", "dentro_da_faixa"
    ]
    with open(os.path.join(a.saida, "resumo.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    print(f"\nResultados salvos em: {a.saida}")


if __name__ == "__main__":
    sys.exit(main())