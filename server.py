from __future__ import annotations
import io, os, tempfile
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter
from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")
PITCHES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
QUALITIES = {
 "": [0,4,7], "m":[0,3,7], "dim":[0,3,6], "aug":[0,4,8], "sus2":[0,2,7], "sus4":[0,5,7],
 "6":[0,4,7,9], "m6":[0,3,7,9], "7":[0,4,7,10], "maj7":[0,4,7,11], "m7":[0,3,7,10],
 "m7b5":[0,3,6,10], "dim7":[0,3,6,9], "add9":[0,2,4,7], "m(add9)":[0,2,3,7],
 "9":[0,2,4,7,10], "maj9":[0,2,4,7,11], "m9":[0,2,3,7,10], "7(b9)":[0,1,4,7,10],
 "11":[0,2,4,5,7,10], "13":[0,2,4,7,9,10]
}
MAJOR_DEGREES=["I","bII","II","bIII","III","IV","bV","V","bVI","VI","bVII","VII"]
MINOR_DEGREES=["i","bII","II","III","#III","iv","bV","v","VI","#VI","VII","#VII"]

@app.get("/")
def index(): return send_from_directory("public", "index.html")

def note_evidence(path, lead_suppression=0.0):
    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < 1024: raise ValueError("区間が短すぎます")
    harmonic, _ = librosa.effects.hpss(y)
    full_cqt = np.abs(librosa.cqt(harmonic, sr=sr, fmin=librosa.note_to_hz("C1"), n_bins=84))
    filtered_cqt = full_cqt.copy()
    lead_bins = []
    if lead_suppression > 0:
        for frame in range(filtered_cqt.shape[1]):
            high = filtered_cqt[36:, frame]
            if high.size < 2: continue
            order = np.argpartition(high, -2)[-2:]
            top_local = int(order[np.argmax(high[order])])
            top, second = float(high[top_local]), float(np.min(high[order]))
            prominence = top / (second + 1e-9)
            if top > np.median(high) * 3 and prominence > 1.18:
                idx = top_local + 36
                amount = min(.98, lead_suppression * min(1.0, .35 + (prominence-1.18)))
                filtered_cqt[max(0,idx-1):min(84,idx+2), frame] *= (1-amount)
                lead_bins.append(idx)
    chroma = np.zeros((12, filtered_cqt.shape[1]))
    for pc in range(12): chroma[pc] = filtered_cqt[pc::12].sum(axis=0)
    energy = np.percentile(chroma, 70, axis=1)
    energy = energy / (energy.max() + 1e-9)
    bass_chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=3)
    bass = np.percentile(bass_chroma, 75, axis=1)
    bass /= bass.max()+1e-9
    spectral_strength=np.percentile(filtered_cqt,70,axis=1)
    spectral_strength/=spectral_strength.max()+1e-9
    octave_notes=[librosa.midi_to_note(24+int(i),unicode=False) for i in np.argsort(spectral_strength)[::-1][:10] if spectral_strength[i]>.16]
    ai_notes=[]
    try:
        from basic_pitch.inference import predict
        _, _, events = predict(path)
        weights=np.zeros(12)
        midi_weights={}
        for ev in events:
            begin,finish,midi,amp,*_=ev
            lead_factor=1.0
            if lead_suppression and int(midi)>=60:
                lead_factor=max(.15,1-lead_suppression*(.35+min(.55,(int(midi)-60)/36)))
            weights[int(midi)%12] += max(.02,finish-begin)*max(.05,float(amp))*lead_factor
            midi_weights[int(midi)]=midi_weights.get(int(midi),0)+max(.02,finish-begin)*max(.05,float(amp))*lead_factor
        if weights.max()>0:
            weights/=weights.max()
            energy=.6*energy+.4*weights
            ai_notes=[PITCHES[i] for i in np.argsort(weights)[::-1][:7] if weights[i]>.18]
            octave_notes=[librosa.midi_to_note(m,unicode=False) for m,_ in sorted(midi_weights.items(),key=lambda item:item[1],reverse=True)[:10]]
    except Exception as exc:
        print("Basic Pitch fallback:", exc)
    lead_notes=[]
    if lead_bins:
        counts=np.bincount(np.array(lead_bins),minlength=84)
        lead_notes=[librosa.midi_to_note(24+int(i),unicode=False) for i in np.argsort(counts)[::-1][:6] if counts[i]>0]
    return energy, bass, ai_notes, lead_notes, octave_notes

def parse_key(key):
    if not key or key=="未指定": return None,None
    root=key.split()[0]
    aliases={"C#":1,"D#":3,"F#":6,"G#":8,"A#":10}
    pc=aliases.get(root, PITCHES.index(root) if root in PITCHES else 0)
    return pc, key.endswith("Minor")

def analyze_chords(energy,bass,key,jazz=True):
    candidates=[]
    bass_pc=int(np.argmax(bass))
    key_pc,is_minor=parse_key(key)
    for root in range(12):
      for quality,intervals in QUALITIES.items():
        if not jazz and quality not in ("","m","7","maj7","m7","dim","sus4"): continue
        mask=np.zeros(12)
        mask[[(root+i)%12 for i in intervals]]=1
        inside=float((energy*mask).sum()/max(1,mask.sum()))
        outside=float((energy*(1-mask)).sum()/max(1,12-mask.sum()))
        coverage=float((energy*mask).sum()/(energy.sum()+1e-9))
        score=.48*inside+.42*coverage-.32*outside+.08*energy[root]+.13*bass[root]-.012*max(0,len(intervals)-4)
        name=PITCHES[root]+quality
        if bass_pc!=root and bass[bass_pc]>.48 and mask[bass_pc]: name+=f"/{PITCHES[bass_pc]}"
        candidates.append((score,name,root,intervals))
    candidates.sort(reverse=True)
    return candidates[:6],bass_pc,key_pc,is_minor

def original_voicing(root, intervals, detected_pitches, bass_pc):
    detected=[]
    for name in detected_pitches:
      try: detected.append(int(round(librosa.note_to_midi(name))))
      except Exception: pass
    center=float(np.median(detected)) if detected else 60
    chosen=[]
    chord_pcs=[(root+i)%12 for i in intervals]
    for pc in chord_pcs:
      matches=[m for m in detected if m%12==pc and m not in chosen]
      if matches: midi=min(matches) if pc==bass_pc else matches[0]
      else:
        options=[m for m in range(36,85) if m%12==pc]
        midi=min(options,key=lambda m:abs(m-center))
      chosen.append(midi)
    bass_matches=[m for m in detected if m%12==bass_pc]
    if bass_pc in chord_pcs and bass_matches:
      bass_midi=min(bass_matches)
      bass_index=chord_pcs.index(bass_pc)
      chosen[bass_index]=bass_midi
      chosen=[m if i==bass_index or m>bass_midi else m+12 for i,m in enumerate(chosen)]
    return [librosa.midi_to_note(m,unicode=False) for m in sorted(chosen)]

def render_lead_reduced(path, strength):
    y, sr = librosa.load(path, sr=22050, mono=True)
    harmonic, percussive = librosa.effects.hpss(y)
    spectrum = librosa.stft(harmonic, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(spectrum)
    frames = max(5, int(sr / 512 * 1.2) | 1)
    stable = np.minimum(magnitude, median_filter(magnitude, size=(1, frames)) * 1.25)
    moving = np.maximum(0, magnitude - stable)
    background_mask = librosa.util.softmask(stable, moving * 1.15, power=2, split_zeros=True)
    base_strength=min(1.0,strength)
    attenuation = 1 - base_strength * (1 - background_mask)
    if strength > 1:
        extra=strength-1
        frequencies=librosa.fft_frequencies(sr=sr,n_fft=2048)
        high_weight=np.clip((frequencies-220)/1800,0,1)[:,None]
        attenuation *= 1-extra*.92*high_weight*(1-.45*background_mask)
    attenuation=np.clip(attenuation,.015,1)
    reduced_harmonic = librosa.istft(magnitude * attenuation * phase, hop_length=512, length=len(y))
    output = reduced_harmonic + percussive
    peak = np.max(np.abs(output))
    if peak > .98: output = output * (.98 / peak)
    memory = io.BytesIO()
    sf.write(memory, output, sr, format="WAV", subtype="PCM_16")
    memory.seek(0)
    return memory

@app.post("/api/lead-preview")
def lead_preview():
    audio=request.files.get("audio")
    if not audio: return jsonify(error="音声がありません"),400
    strength=max(0,min(2,float(request.form.get("lead_suppression","55"))/100))
    tmp=None
    try:
      with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        audio.save(f); tmp=f.name
      return send_file(render_lead_reduced(tmp,strength),mimetype="audio/wav",download_name="lead-reduced.wav")
    except Exception as exc: return jsonify(error=str(exc)),500
    finally:
      if tmp and os.path.exists(tmp): os.unlink(tmp)

@app.post("/api/analyze")
def analyze():
    audio=request.files.get("audio")
    if not audio: return jsonify(error="音声がありません"),400
    key=request.form.get("key","未指定")
    jazz=request.form.get("jazz","true")=="true"
    lead_suppression=max(0,min(2,float(request.form.get("lead_suppression","0"))/100))
    tmp=None
    try:
      with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        audio.save(f); tmp=f.name
      energy,bass,ai_notes,lead_notes,octave_notes=note_evidence(tmp,lead_suppression)
      choices,bass_pc,key_pc,is_minor=analyze_chords(energy,bass,key,jazz)
      top=choices[0]
      margin=max(0,top[0]-choices[1][0]) if len(choices)>1 else .15
      conf=round(min(96,55+margin*240))
      degree="—" if key_pc is None else (MINOR_DEGREES if is_minor else MAJOR_DEGREES)[(top[2]-key_pc)%12]
      notes=[PITCHES[(top[2]+i)%12] for i in top[3]]
      voicing=original_voicing(top[2],top[3],octave_notes,bass_pc)
      detected=[PITCHES[i] for i in np.argsort(energy)[::-1][:8] if energy[i]>.16]
      return jsonify(chord=top[1],degree=degree,bass=PITCHES[bass_pc],notes=notes,voicing=voicing,detected_notes=detected,
        lead_notes=lead_notes,detected_pitches=octave_notes,lead_suppression=round(lead_suppression*100),
        ai_notes=ai_notes,confidence=conf,
        alternatives=[{"chord":x[1],"notes":[PITCHES[(x[2]+iv)%12] for iv in x[3]],"voicing":original_voicing(x[2],x[3],octave_notes,bass_pc),"confidence":max(20,round(conf-(i+1)*5-(top[0]-x[0])*110))} for i,x in enumerate(choices[1:])],
        explanation="Basic Pitchの音高推定、打楽器を抑えたクロマ、低域のベース候補を統合したローカル解析です。")
    except Exception as exc: return jsonify(error=str(exc)),500
    finally:
      if tmp and os.path.exists(tmp): os.unlink(tmp)

if __name__=="__main__":
    print("Chord Scope v9: http://localhost:3000")
    app.run(host="127.0.0.1",port=3000,debug=False)
