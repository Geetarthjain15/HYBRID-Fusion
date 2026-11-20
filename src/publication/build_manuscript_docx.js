const fs = require('fs');
const path = require('path');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,Table,TableRow,TableCell,
  WidthType,BorderStyle,ShadingType,ImageRun,Header,Footer,PageNumber,PageBreak} = require('docx');

const root=path.resolve(process.argv[2]||'.');
const src=path.join(root,'publication_results/manuscript_draft/AGMF_Net_Manuscript.md');
const out=path.join(root,'publication_results/manuscript_draft/AGMF_Net_Manuscript.docx');
const lines=fs.readFileSync(src,'utf8').split(/\r?\n/); const children=[];
const clean=s=>s.replace(/\*\*/g,'').replace(/\$/g,'').replace(/`([^`]+)`/g,'$1').replace(/\[([^\]]+)\]\(([^)]+)\)/g,'$1 ($2)');
function para(text,opts={}){return new Paragraph({spacing:{after:opts.after??120,line:300},alignment:opts.align,
  children:[new TextRun({text:clean(text),bold:opts.bold,italics:opts.italics,size:opts.size||21,font:'Arial'})]});}
function imageFor(rel,caption){const f=path.join(path.dirname(src),rel); if(!fs.existsSync(f)) return para(`[Missing figure: ${rel}]`);
  const data=fs.readFileSync(f); return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:80},
    children:[new ImageRun({data,type:'png',transformation:{width:610,height:Math.round(610*(caption.includes('architecture')?8/13:.58))},altText:{title:caption,description:caption,name:caption}})]});}
function tableFrom(rows){const cells=rows[0].length,w=Math.floor(9360/cells); return new Table({width:{size:9360,type:WidthType.DXA},columnWidths:Array(cells).fill(w),
  rows:rows.map((r,ri)=>new TableRow({children:r.map(x=>new TableCell({width:{size:w,type:WidthType.DXA},shading:ri===0?{fill:'DCE6F1',type:ShadingType.CLEAR}:undefined,
    margins:{top:70,bottom:70,left:70,right:70},children:[para(x,{bold:ri===0,size:17,after:0})]}))}))});}
for(let i=0;i<lines.length;){let l=lines[i].trim(); if(!l){i++;continue;}
  const im=l.match(/^!\[([^\]]*)\]\(([^)]+)\)/); if(im){children.push(imageFor(im[2],im[1]));i++;continue;}
  if(l.startsWith('|')){const rows=[];while(i<lines.length&&lines[i].trim().startsWith('|')){const r=lines[i].trim().split('|').slice(1,-1).map(x=>clean(x.trim()));if(!r.every(x=>/^[-:]+$/.test(x)))rows.push(r);i++;}if(rows[0]&&rows[0][0]==='Variant')children.push(new Paragraph({children:[new PageBreak()]}));children.push(tableFrom(rows));children.push(para(''));continue;}
  const h=l.match(/^(#{1,3})\s+(.*)/); if(h){const level=h[1].length===1?HeadingLevel.TITLE:h[1].length===2?HeadingLevel.HEADING_1:HeadingLevel.HEADING_2;
    if(h[2]==='References')children.push(new Paragraph({children:[new PageBreak()]}));
    children.push(new Paragraph({text:clean(h[2]),heading:level,spacing:{before:level===HeadingLevel.TITLE?0:240,after:120},alignment:level===HeadingLevel.TITLE?AlignmentType.CENTER:undefined}));i++;continue;}
  if(/^[-*]\s+/.test(l)){children.push(new Paragraph({text:clean(l.replace(/^[-*]\s+/,'')),bullet:{level:0},spacing:{after:80,line:280}}));i++;continue;}
  if(/^\d+\.\s+/.test(l)){children.push(new Paragraph({text:clean(l.replace(/^\d+\.\s+/,'')),numbering:{reference:'numbered',level:0},spacing:{after:80,line:280}}));i++;continue;}
  if(l.startsWith('> ')){children.push(para(l.slice(2),{italics:true}));i++;continue;}
  let block=l;i++;while(i<lines.length&&lines[i].trim()&&!/^(#|\||!\[|[-*]\s+|\d+\.\s+)/.test(lines[i].trim())){block+=' '+lines[i].trim();i++;}
  children.push(para(block));
}
const doc=new Document({creator:'AGMF-Net research team',title:'AGMF-Net manuscript',description:'Editable submission draft',
  numbering:{config:[{reference:'numbered',levels:[{level:0,format:'decimal',text:'%1.',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:420,hanging:240}}}}]}]},
  styles:{default:{document:{run:{font:'Arial',size:21},paragraph:{spacing:{line:300}}}},paragraphStyles:[
    {id:'Title',name:'Title',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Arial',size:32,bold:true,color:'17365D'},paragraph:{alignment:AlignmentType.CENTER,spacing:{after:260},outlineLevel:0}},
    {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Arial',size:26,bold:true,color:'17365D'},paragraph:{spacing:{before:260,after:120},outlineLevel:0}},
    {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Arial',size:22,bold:true,color:'365F91'},paragraph:{spacing:{before:200,after:100},outlineLevel:1}}]},
  sections:[{properties:{page:{margin:{top:1008,right:1008,bottom:1008,left:1008},size:{width:11906,height:16838}}},
    headers:{default:new Header({children:[new Paragraph({alignment:AlignmentType.RIGHT,children:[new TextRun({text:'AGMF-Net — editable submission draft',size:16,color:'64748B'})]})]})},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:'Page ',size:16}),new TextRun({children:[PageNumber.CURRENT],size:16})]})]})},children}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(out,b);console.log(out);});
