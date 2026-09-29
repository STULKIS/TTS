// Real page + schema with a fake HTTP speech response; verifies controls, not voice quality.
const {JSDOM, VirtualConsole} = require('jsdom');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.join(__dirname, '..');
const schema = JSON.parse(execFileSync(process.env.PYTHON || 'python3', ['-c',
  "import sys,json;sys.path.insert(0,'tools');import voice_design as v;print(json.dumps({'fields':v.SCHEMA,'languages':v.LANGUAGES,'engine':'qwen-design','available':True}))"], {cwd:root, encoding:'utf8'}));
const source = fs.readFileSync(path.join(root, 'tools/type_ui.py'), 'utf8');
const html = source.split('PAGE_HTML = r"""')[1].split('"""')[0].replace('__PACK__', '[]');
const sent = [];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => errors.push(e.message));
const dom = new JSDOM(html, {url:'http://designer.test/', runScripts:'dangerously', virtualConsole,
  beforeParse(w) {
    w.URL.createObjectURL = () => 'blob:design-test'; w.URL.revokeObjectURL = () => {};
    w.HTMLMediaElement.prototype.play = () => Promise.resolve();
    w.fetch = async (url, opts) => {
      if (url === '/api/design/schema') return {ok:true,json:async()=>schema};
      if (url === '/api/design/preview') {
        sent.push({preview:JSON.parse(opts.body)});
        return {ok:true,json:async()=>({instruction:'Exact preview test'})};
      }
      if (url === '/api/tts') {
        sent.push(JSON.parse(opts.body));
        return {ok:true,headers:{get:()=> '1'},blob:async()=>new w.Blob(['test-only'])};
      }
      return {ok:true,json:async()=>url === '/api/presets' ?
        {presets:[{no:1,name:'Warm',description:'Warm dry adult voice.',class:'Healer',gender:'M',pitch:'Low',pace:'Tale',rarity:3,pitch_shift:-3,speed:.95}], classes:['Healer'],count:1} :
        url === '/api/pack' ? {items:[]} : {version:'TEST',languages:['en','pt'],pack:0,presets:1,cpu_threads:6}};
    };
  },
});
const w = dom.window;
const $ = id => w.document.getElementById(id);
const design = (name, scope=$('design-fields')) => scope.querySelector(`[data-design="${name}"]`);
(async () => {
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(errors,[]);
  assert.equal($('tab-design').classList.contains('on'),true);
  assert.equal($('design-fields').querySelectorAll('[data-design]').length,62);
  assert.equal($('tab-preset').disabled,true);
  assert.equal($('speed').disabled,true);
  assert.equal($('splitm').value,'cut0');
  assert.equal($('design-edit').disabled,false);
  assert.equal([...$('textlang').options].some(o=>o.value==='pt'),true);
  assert.equal([...$('textlang').options].some(o=>o.value==='yue'),false);
  $('plist').querySelector('.prow').click();
  assert.equal(design('description').value,'Warm dry adult voice.');
  assert.equal($('pitch').value,'0', 'preset description must not sneak in DSP pitch');
  design('roughness').value='slightly raspy'; design('whisper').value='whispered speech';
  design('direction').value='Start confident, then sound worried.';
  design('crying').value='tearful voice';
  $('text').value='Say these words only.';
  await $('go').onclick();
  assert.equal(sent[0].text,'Say these words only.');
  assert.equal(sent[0].engine,'qwen-design');
  assert.equal(sent[0].design.roughness,'slightly raspy');
  assert.equal(sent[0].design.direction,'Start confident, then sound worried.');
  assert.equal(sent[0].ref_audio_path,null);
  assert.equal(sent[0].fx,'');
  await $('previewdesign').onclick();
  assert.equal(sent[1].preview.design.crying,'tearful voice');
  assert.equal($('design-preview').textContent,'Exact preview test');

  $('setupname').value='My designed voice'; $('savesetup').click();
  design('roughness').value='gravelly'; $('loadsetup').click();
  assert.equal(design('roughness').value,'slightly raspy');
  assert.equal($('speed').value,'1');
  $('scriptq').value='First.\nSecond.'; $('prepare').click();
  const rows = [...$('queue').children];
  assert.equal(rows.length,2);
  assert.equal(rows[0].querySelectorAll('[data-design]').length,62);
  design('emotion',rows[0]).value='angry'; design('emotion',rows[1]).value='happy';
  design('emotion').value='sad'; // main controls must not overwrite prepared rows
  await $('brender').onclick();
  assert.equal(sent[2].design.emotion,'angry');
  assert.equal(sent[3].design.emotion,'happy');
  assert.equal(sent[2].engine,'qwen-design');
  assert.equal(sent[2].ref_audio_path,undefined);
  assert.deepEqual(errors,[]);
  dom.window.close();
  console.log('PASS: 58 trait axes + acting/brief controls reach single and per-line requests; profiles round-trip; no reference or FX substitution');
})().catch(e=>{console.error(e);dom.window.close();process.exitCode=1;});
