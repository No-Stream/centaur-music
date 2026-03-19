# ji_chorale_v2 — Ideas & Future Directions

Discussion from 2026-03-19. Roughly priority-ordered within each section.

---

## Rhythmic innovations

**Suspension and delay in bass / upper voices** ← *implemented v2.1*
A prepared dissonance held from one chord into the next. Baroque chorales use
7-6, 4-3, and 9-8 suspensions constantly. In this piece the most natural target
is the alto: when moving from vi (Fs3/A3/Cs4) to iv (D3/F3/A3), the alto holds
Cs4 for one beat over D3 — a major-7th clash — before resolving to A3. Also
tractable: a bass tritone suspension at the Development's F#m7→Bm transition
(Fs2 held over B2 = tritone = diabolus in musica).

**Hemiola at climaxes**
Group 4/4 in groups of 3 for 1-2 bars before a big downbeat resolution. In the
lead melody, accent every 3rd eighth note approaching bar 16 so the meter feels
like it's fighting itself before the D5→A5 leap.

**Contrary motion between counter and lead**
Currently they tend to rest and phrase at similar points. Having the counter run
in short eighth-note bursts exactly while the lead holds a long note (and vice
versa) creates far more textural life. The voices should fill each other's
silences.

**Pickup anticipations in the lead**
Several melodic arrivals land flat on the bar. Shifting some to arrive a
half-beat early, or to approach from below on a weak eighth, makes them feel
more propulsive.

**Rhythmic augmentation in the Ending**
Make each successive section of the Ending (vi / Dm7 / Amaj7) progressively
slower — half-note movement becoming whole-note movement — to create a
deliberate sense of time dilating toward the end.

---

## Baroque ornamentations

**Baroque arpeggiation in the alto** ← *implemented v2.1*
At section entries (B section bar 19, Reprise bar 36), the alto rolls upward
through chord tones in sixteenth notes (Fs3→A3→Cs4 or A3→Cs4) before holding.
NOT the cliché "arpeggiate every chord" synth move — used sparingly at
structural arrivals only.

**Trills on leading tones**
PitchMotionSpec.vibrato at high rate on Gs4 (leading tone) before A4 arrivals.
Could also use very rapid two-note alternation phrases if more literal trill is
wanted.

**Appoggiatura (leaning note)**
A short note a step above on the beat, resolving immediately down. E.g., B4(E)
→ A4(long) — B4 takes the beat emphasis and resolves. Trivially added before
any melodic arrival with score.add_note.

**Mordent**
A(E) → Gs4(E) → A4(Q) in rapid triplet eighths (three 16th-note triplets)
before a main note. At 88 BPM these are fast enough to feel ornamental.

**Tierce de Picardie at final cadence**
At a vi→I resolution, instead of the expected minor/natural resolution, end
on the major third — Cs4 (natural major) over A root. Already partly touched
in the Amaj7 moment but could be made more structurally significant.

**Echo dynamics (terraced)**
Counter echoes the last 2 notes of a lead phrase 1 beat later, at lower
velocity. Baroque "forte/piano" effect, very characteristic.

---

## Making climaxes more emotive

**Climax preparation — restraint before bar 16** ← *implemented v2.1*
Remove the Cs5 preview in bar 13 (was: B4-Cs5-B4, now: B4-A4-B4). Keep the
melody circling below until bar 15's Cs5 is the last pre-climax step, making
bar 16's D5→A5 leap feel genuinely earned. Also pull back velocity in bars
12-13 so the dynamic ramp is steeper going into bar 16.

**Tutti convergence at the climax**
At the bar-16 peak specifically, override stagger to 0 so all voices land
together. The accumulated stagger texture suddenly snapping tight at the
climax moment is visceral.

**Subharmonic reinforcement**
At A5 (bar 16), double the bass with a very soft A1 or drop-octave A2 hit.
Sub-bass confirmation of the tonic at the arc's peak = organ/orchestral effect.

**Delayed resolution after the climax**
Let A5 hang over a harmony that doesn't yet support it (bass on D3 while lead
holds A5) before the harmony finally catches up. The unresolved high note is
more powerful than the climax note itself.

**Second climax in the Ending — Gs4 over Dm7**
Bar 49's Gs4 over Dm7 (bittersweet, maj7-ish clash) is the emotional second
peak but the lead currently just holds it passively. Have the counter rush
upward in response, or sustain a different dissonance, to acknowledge it.

---

## Structure / extension

**Fugal exposition (point of imitation in B section)** ← *implemented v2.1*
Lead opens the B section with a figure; counter enters 1 bar later transposed
a 4th below (tonal answer = dominant relationship). Not a full fugue — a
baroque point of imitation. Uses B3/Gs4 as the answer pitches since B is the
dominant of the key of E (V chord territory). Creates genuine 2-voice
counterpoint texture in the brightest section.

**Coda / dissolution** ← *implemented v2.1*
After the final Amaj7, the piece fragments: lead plays isolated notes with long
silences, counter has one last falling phrase (E4→D4→Cs4→A3), bass sustains a
soft tonic. Silence becomes part of the form. ~13-15 seconds.

**Second Development with wider harmonic range**
Current Development stays in close keys (F#m / Bm / E). A more Bach-like
Development would tonicize C# minor (Cs4 as a local root) or even Bb — both
reachable by 5-limit comma paths from A — before navigating home. The JI
character shines here: the tuning shifts in distant keys are audible and
beautiful.

**Fugal stretto before climax**
Before bar 16, counter and lead both play the opening A4-Cs5 motif but with
the counter entering only a beat behind the lead — a tight canon. Then they
diverge into the climax. Creates a sense of the voices being pulled irresistibly
toward the peak.

**True pianissimo section**
The piece never truly goes quiet after the A section begins. A lull — perhaps
inside the Development — where texture thins to bass + lead only at low dynamic
would make the next full-ensemble entry feel like a genuine event.
