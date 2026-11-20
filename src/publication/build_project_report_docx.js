const fs=require('fs'),path=require('path');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,AlignmentType,ImageRun,Header,Footer,
  PageNumber,PageBreak,ExternalHyperlink,TableOfContents,LevelSuffix}=require('docx');
const root=path.resolve(process.argv[2]||'.');
const src=path.join(root,'publication_results/project_report/AGMF_Net_Project_Report.md');
const out=path.join(root,'publication_results/project_report/AGMF_Net_Project_Report.docx');
const lines=fs.readFileSync(src,'utf8').split(/\r?\n/), children=[];
const strip=s=>s.replace(/\*\*/g,'').replace(/`([^`]+)`/g,'$1');
function runsWithLinks(s,opts={}){s=strip(s);const re=/(https?:\/\/\S+)/g,parts=s.split(re),runs=[];for(const x of parts){if(!x)continue;if(/^https?:/.test(x)){runs.push(new ExternalHyperlink({link:x.replace(/[.,;]$/,''),children:[new TextRun({text:x,color:'0563C1',underline:{},font:'Times New Roman',size:24})]}));}else runs.push(new TextRun({text:x,font:'Times New Roman',size:opts.size||24,bold:opts.bold,italics:opts.italics}));}return runs;}
function p(text,opts={}){return new Paragraph({alignment:opts.align||AlignmentType.JUSTIFIED,spacing:{after:opts.after??120,line:360},indent:opts.indent?{firstLine:360}:undefined,children:runsWithLinks(text,opts)});}
function fig(rel,title){const f=path.resolve(path.dirname(src),rel),data=fs.readFileSync(f);let h=360;if(rel.includes('architecture'))h=375;if(rel.includes('gradcam'))h=330;return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:150,after:100},children:[new ImageRun({data,type:'png',transformation:{width:600,height:h},altText:{title,description:title,name:title}})]});}
let titleMode=true,inRefs=false;
for(let i=0;i<lines.length;){let l=lines[i].trim();if(!l){i++;continue;}
 const im=l.match(/^!\[([^\]]*)\]\(([^)]+)\)/);if(im){children.push(fig(im[2],im[1]));i++;continue;}
 const h=l.match(/^(#{1,3})\s+(.*)/);if(h){const t=strip(h[2]);if(t==='Contents'){children.push(new Paragraph({children:[new PageBreak()]}));titleMode=false;children.push(new Paragraph({text:'TABLE OF CONTENTS',heading:HeadingLevel.HEADING_1,alignment:AlignmentType.CENTER}));children.push(new TableOfContents('Contents',{hyperlink:true,headingStyleRange:'1-3'}));children.push(new Paragraph({children:[new PageBreak()]}));i++;continue;}
  if(t==='5. References')inRefs=true;
  const level=h[1].length===1?(titleMode?HeadingLevel.TITLE:HeadingLevel.HEADING_1):h[1].length===2?HeadingLevel.HEADING_2:HeadingLevel.HEADING_3;
  children.push(new Paragraph({text:t,heading:level,alignment:level===HeadingLevel.TITLE?AlignmentType.CENTER:undefined,spacing:{before:level===HeadingLevel.TITLE?100:220,after:140}}));i++;continue;}
 if(/^\d+\.\s+/.test(l)){children.push(new Paragraph({children:runsWithLinks(l.replace(/^\d+\.\s+/,'')),numbering:{reference:inRefs?'references':'numbered',level:0},spacing:{after:90,line:360}}));i++;continue;}
 if(l.startsWith('> ')){children.push(p(l.slice(2),{italics:true,indent:true}));i++;continue;}
 let block=l;i++;while(i<lines.length&&lines[i].trim()&&!/^(#|!\[|\d+\.\s+)/.test(lines[i].trim())){block+=' '+lines[i].trim();i++;}
 children.push(p(block,{align:titleMode?AlignmentType.CENTER:AlignmentType.JUSTIFIED,indent:!titleMode}));
}
const doc=new Document({creator:'AGMF-Net project team',title:'AGMF-Net B.Tech Project Report',
 numbering:{config:[{reference:'numbered',levels:[{level:0,format:'decimal',text:'%1.',suffix:LevelSuffix.SPACE,alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:300}}}}]},
  {reference:'references',levels:[{level:0,format:'decimal',text:'%1.',suffix:LevelSuffix.SPACE,alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:300}}}}]}]},
 styles:{default:{document:{run:{font:'Times New Roman',size:24},paragraph:{spacing:{line:360,after:120}}}},paragraphStyles:[
  {id:'Title',name:'Title',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Times New Roman',size:34,bold:true},paragraph:{alignment:AlignmentType.CENTER,spacing:{before:120,after:220},outlineLevel:0}},
  {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Times New Roman',size:28,bold:true},paragraph:{spacing:{before:260,after:140},outlineLevel:0}},
  {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Times New Roman',size:26,bold:true},paragraph:{spacing:{before:220,after:120},outlineLevel:1}},
  {id:'Heading3',name:'Heading 3',basedOn:'Normal',next:'Normal',quickFormat:true,run:{font:'Times New Roman',size:24,bold:true,italics:true},paragraph:{spacing:{before:180,after:100},outlineLevel:2}}]},
 sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
  headers:{default:new Header({children:[new Paragraph({alignment:AlignmentType.RIGHT,children:[new TextRun({text:'AGMF-Net Project Report',font:'Times New Roman',size:18})]})]})},
  footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:'Page ',font:'Times New Roman',size:20}),new TextRun({children:[PageNumber.CURRENT],font:'Times New Roman',size:20})]})]})},children}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(out,b);console.log(out)});
