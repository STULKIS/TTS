// Full-page DOM behavior tests. Fetch/audio are test doubles, not voice synthesis.
// NODE_PATH=<jsdom install>/node_modules node tests/studio_controls_dom.cjs
const {JSDOM, VirtualConsole} = require('jsdom');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../tools/type_ui.py'), 'utf8');
const pack = [
  {char:'dragon', lang:'en', clip:'en.wav', transcript:'Dragon reference.', seconds:4},
  {char:'drake', lang:'ja', clip:'ja.wav', transcript:'Drake reference.', seconds:5},
];
const presets = Array.from({length:30}, (_, i) => ({no:i+1, name:'Warm voice '+i,
  class:'Healer', gender:i%2 ? 'M' : 'F', pitch:'Low', pace:'Tale', rarity:3,
  description:'Warm reference brief.', pitch_shift:-3, speed:.95, signature:''}));
const html = source.split('PAGE_HTML = r"""')[1].split('"""')[0].replace('__PACK__', JSON.stringify(pack));
const errors = [];
const consoleCapture = new VirtualConsole();
consoleCapture.on('jsdomError', e => errors.push(e.message));
const requests = [];
const revoked = [];
let failNext = false;
let urlId = 0;
const dom = new JSDOM(html, {
  url:'http://studio.test/', runScripts:'dangerously', virtualConsole:consoleCapture,
  beforeParse(w) {
    w.URL.createObjectURL = () => 'blob:test-' + (++urlId);
    w.URL.revokeObjectURL = url => revoked.push(url);
    w.HTMLMediaElement.prototype.play = () => Promise.resolve();
    w.fetch = async (url, opts) => {
      if (url === '/api/tts') {
        requests.push(JSON.parse(opts.body));
        if (failNext) { failNext=false; return {ok:false, json:async () => ({message:'test failure'})}; }
        return {ok:true, headers:{get: k => k === 'X-Take-Cache' ? 'miss' : '1.0'},
          blob:async () => new w.Blob(['test-only-audio'])};
      }
      const data = url === '/api/pack' ? {items:pack} : url === '/api/presets' ?
        {presets, classes:['Healer'], count:30} :
        {version:'TEST-DOUBLE', languages:['en','ja'], pack:2, presets:30, cpu_threads:6};
      return {ok:true, json:async () => data};
    };
  },
});
const w = dom.window;
const $ = id => w.document.getElementById(id);
const rows = () => [...$('queue').children];
const input = (row, key) => row.querySelector(`[data-control="${key}"]`);
const clickText = (row, text) => [...row.querySelectorAll('button')].find(b => b.textContent === text).onclick();
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(errors, []);
  assert.equal($('expr').tagName, 'FIELDSET', 'sampling must not be hidden in closed details');
  assert.equal($('temperature').type, 'number', 'current value must be visible');
  assert.equal($('plist').children.length, 24);
  $('morepresets').click();
  assert.equal($('plist').children.length, 30, 'every catalog row reachable');
  $('genderfilter').value = 'M'; $('genderfilter').onchange();
  assert.equal($('plist').children.length, 15);
  $('genderfilter').value = 'all'; $('genderfilter').onchange();
  assert.equal($('perf').value, '');
  $('setupname').value = 'Dry narrator'; $('designnote').value = 'Mature, dry (reference needed)';
  $('pitch').value = '-2'; $('speed').value = '.9'; $('temperature').value = '.8';
  $('savesetup').click();
  $('pitch').value = '5'; $('temperature').value = '1.2';
  $('loadsetup').click();
  assert.equal($('pitch').value, '-2'); assert.equal($('temperature').value, '0.8');
  assert.equal($('perf').value, '', 'saving/loading settings must not invent a voice');
  const saved = JSON.parse(w.localStorage.getItem('gacha.delivery.setups.v1'));
  assert.equal(saved['Dry narrator'].controls.pitch, -2);
  assert.equal(saved['Dry narrator'].note, 'Mature, dry (reference needed)');
  assert.equal('text' in saved['Dry narrator'].controls, false);

  $('scriptq').value = 'First line.\nSecond line.';
  $('prepare').click();
  assert.equal(rows().length, 2);
  assert.equal(rows()[0].querySelectorAll('[data-control]').length, 14);
  assert.equal(rows()[0].querySelectorAll('[data-fx]').length, 10);
  input(rows()[0], 'pitch').value = '3'; input(rows()[0], 'temperature').value = '.7';
  input(rows()[1], 'pitch').value = '-4'; input(rows()[1], 'speed_factor').value = '1.2';
  input(rows()[1], 'text_lang').value = 'ja';
  rows()[1].querySelector('select').value = 'pack:1';
  rows()[1].querySelector('textarea').value = 'Edited second line.';
  $('pitch').value = '10'; $('temperature').value = '1.5';
  await $('brender').onclick();
  assert.equal(requests.length, 2);
  assert.equal(requests[0].pitch, 3); assert.equal(requests[0].temperature, .7);
  assert.equal(requests[1].pitch, -4); assert.equal(requests[1].speed_factor, 1.2);
  assert.equal(requests[1].temperature, .8, 'top changes do not overwrite line snapshot');
  assert.equal(requests[1].text, 'Edited second line.');
  assert.equal(requests[1].text_lang, 'ja');
  assert.equal(requests[1].ref_audio_path, 'drake/ja.wav');
  assert.equal(requests[1].prompt_text, 'Drake reference.');
  assert.equal($('brender').disabled, false);
  assert.equal(rows()[0].querySelector('fieldset').disabled, false);
  assert.equal(rows()[0].querySelector('audio').style.display, '');

  rows()[0].querySelector('select').value = 'perf:cheer-dragon';
  input(rows()[0], 'seed').value = '42';
  await clickText(rows()[0], 'Fresh take');
  assert.equal(requests.at(-1).seed, -1);
  assert.equal(requests.at(-1).reuse_take, false);
  assert.equal(requests.at(-1).ref_audio_path, 'perf:cheer-dragon');
  assert.equal(requests.at(-1).prompt_lang, 'en');
  assert.ok(revoked.length > 0, 'replaced audio URL is released');

  const count = requests.length;
  input(rows()[1], 'speed_factor').value = '9';
  await clickText(rows()[1], 'Render this line');
  assert.equal(requests.length, count, 'invalid controls must not reach the engine even while fieldset is locked');
  assert.match(rows()[1].textContent, /highlighted line control/);
  input(rows()[1], 'speed_factor').value = '1';
  failNext = true;
  await clickText(rows()[1], 'Render this line');
  assert.match(rows()[1].textContent, /test failure/);
  assert.equal(rows()[1].querySelector('fieldset').disabled, false);
  await clickText(rows()[1], 'Render this line');
  assert.doesNotMatch(rows()[1].querySelector('[role="status"]').textContent, /failed/);

  // Do not silently truncate scripts or destroy previously edited rows.
  $('scriptq').value = Array(21).fill('Text').join('\n'); $('prepare').click();
  assert.equal(rows().length, 2);
  assert.match($('msg').textContent, /nothing was discarded/);
  $('savedsetup').value = 'Dry narrator'; $('deletesetup').click();
  assert.equal(Object.keys(JSON.parse(w.localStorage.getItem('gacha.delivery.setups.v1'))).length, 0);
  assert.deepEqual(errors, []);
  dom.window.close();
  console.log('PASS: full-page controls, catalog, setup persistence, per-row payloads, reroll and error recovery');
})().catch(e => { console.error(e); dom.window.close(); process.exitCode=1; });
