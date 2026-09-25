/* Speaker timing strips. Times and viewport bounds are in seconds. */
(function () {
  "use strict";

  const PALETTE = ["#4fa8e8", "#ef9765", "#73c994", "#c69aef", "#e57dac",
    "#55c5cc", "#d5b554", "#8eacef", "#c7bd77", "#b4cf7b", "#d28a80", "#97b9b5"];

  function speakerColour(label) {
    let hash = 2166136261;
    for (const character of label) {
      hash ^= character.codePointAt(0);
      hash = Math.imul(hash, 16777619);
    }
    return PALETTE[(hash >>> 0) % PALETTE.length];
  }

  function timestamp(value) {
    const tenths = Math.round(value * 10);
    const hours = Math.floor(tenths / 36000);
    const minutes = Math.floor(tenths / 600) % 60;
    const seconds = ((tenths % 600) / 10).toFixed(1).padStart(4, "0");
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${seconds}` : `${minutes}:${seconds}`;
  }

  function seconds(value) {
    if (typeof value !== "number" && (typeof value !== "string" || !value.trim())) return NaN;
    return Number(value);
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  class WaveformSpeakers {
    constructor(host) {
      this.host = host;
      this.host.classList.add("waveformSpeakers");
      this.tracks = [];
      this.viewport = null;
      this.clear();
    }

    clear() {
      this.tracks = [];
      this.viewport = null;
      this.host.replaceChildren();
      this.host.hidden = true;
    }

    setTracks(tracks) {
      this.tracks = [];
      this.host.replaceChildren();
      for (const track of Array.isArray(tracks) ? tracks : []) {
        if (!track || !Array.isArray(track.segments)) continue;
        const speakers = new Map();
        for (const segment of track.segments) {
          if (!segment || typeof segment !== "object") continue;
          const start = Math.max(0, seconds(segment.start));
          const end = seconds(segment.end);
          if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
          const name = String(segment.speaker ?? "").trim() || "Unknown speaker";
          // Silence is a gap, not a speaker lane or a legend entry.
          if (name.toUpperCase() === "SILENCE" || segment.is_silence_marker === true) continue;
          if (!speakers.has(name)) speakers.set(name, []);
          speakers.get(name).push({start, end});
        }
        if (!speakers.size) continue;

        const label = String(track.label ?? "").trim() || "Speakers";
        const group = element("div", "waveformSpeakerGroup");
        group.setAttribute("role", "group");
        group.setAttribute("aria-label", `${label}: speaker timing`);
        const heading = element("div", "waveformSpeakerHeading");
        heading.append(element("span", "waveformSpeakerTrackLabel", label));
        const legend = element("div", "waveformSpeakerLegend");
        legend.setAttribute("aria-label", "Speakers in lane order");
        const laneArea = element("div", "waveformSpeakerLanes");
        laneArea.tabIndex = 0;
        laneArea.setAttribute("role", "region");
        laneArea.setAttribute("aria-label", `${label} speaker lanes. Scroll vertically to see all speakers.`);
        const lanes = [];
        const names = [...speakers.keys()].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
        if (names.length > 7) heading.append(element("span", "waveformSpeakerScrollHint", "Scroll lanes ↕"));
        for (let index = 0; index < names.length; index++) {
          const name = names[index];
          const colour = speakerColour(name);
          const intervals = speakers.get(name).sort((a, b) => a.start - b.start || a.end - b.end);
          // Duplicate and overlapping segments from the same speaker form one line.
          const merged = [];
          for (const interval of intervals) {
            const previous = merged[merged.length - 1];
            if (previous && interval.start <= previous.end) previous.end = Math.max(previous.end, interval.end);
            else merged.push({...interval});
          }
          const item = element("span", "waveformSpeakerLegendItem");
          item.style.setProperty("--speaker-color", colour);
          item.title = `Lane ${index + 1}: ${name}`;
          const swatch = element("span", "waveformSpeakerSwatch");
          swatch.setAttribute("aria-hidden", "true");
          item.append(swatch, element("span", "waveformSpeakerName", `${index + 1}. ${name}`));
          legend.append(item);

          const node = element("div", "waveformSpeakerLane");
          node.style.setProperty("--speaker-color", colour);
          node.setAttribute("role", "group");
          node.setAttribute("aria-label", `Lane ${index + 1}: ${name}`);
          node.title = `Lane ${index + 1}: ${name}`;
          laneArea.append(node);
          lanes.push({name, intervals: merged, node});
        }
        heading.append(legend);
        group.append(heading, laneArea);
        this.host.append(group);
        this.tracks.push({label, lanes});
      }
      this.render();
    }

    setViewport(start, end) {
      this.viewport = Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start
        ? {start, end} : null;
      this.render();
    }

    render() {
      this.host.hidden = !this.tracks.length || !this.viewport;
      for (const track of this.tracks) {
        for (const lane of track.lanes) {
          const fragment = document.createDocumentFragment();
          if (this.viewport) {
            const {start, end} = this.viewport;
            const duration = end - start;
            for (const interval of lane.intervals) {
              if (interval.start >= end) break;
              if (interval.end <= start) continue;
              const left = Math.max(start, interval.start);
              const right = Math.min(end, interval.end);
              const line = element("span", "waveformSpeakerLine");
              line.style.left = `${(left - start) / duration * 100}%`;
              line.style.width = `${(right - left) / duration * 100}%`;
              const description = `${track.label}, ${lane.name}: ${timestamp(interval.start)} to ${timestamp(interval.end)}`;
              line.title = description;
              line.setAttribute("role", "img");
              line.setAttribute("aria-label", description);
              fragment.append(line);
            }
          }
          lane.node.replaceChildren(fragment);
        }
      }
    }
  }

  window.WaveformSpeakers = WaveformSpeakers;
})();
