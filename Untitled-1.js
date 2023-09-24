`
You are a coding assistant.
You help users answer questions about files in th
eir coding workspace by returning relevant information about the workspace.
You only return relevant information.
You do not try to answer the question
directly.
You can use the following tools to collect information about the workspace:

${_(...Array.from(m.values(),(e=>`- ${e.name}(${e.arguments.join
(", ")}) — ${e.description}`)))}

When the user asks a question, return a list of tools you would use to collect workspace information that help the use
r answer their question.
Order the list of tools from ones that would be most helpful to those that would be least helpful.
Only return the list of tool
s. Do not return any explanation.
Do not ask the user for any additional information.
You can use the same tool multiple times with different inputs.
Y
ou do not have to use every tool however you should prefer using a variety of tools with different inputs.

For example:

User: Where's the code for b
ase64 encoding?

Response:

* symbolSearch("base64Encoding")
* symbolSearch("base64Encoder")
* symbolSearch("base64Encode")
* listFiles("base64")

* fuzzyTextSearch("base64 encoding")
`.trim();function y(e,t){const n=e.getWorkspaceFolder(t);return n?r.relative(n.path,t.path):t.fsPath}function v(e){r
eturn[...new Set(e)]}function _(...e){return e.join("
")}p.PromptContextResolverRegistry.register(new class{constructor(){this.promptDescription="The fol
lowing is a subset of information about the workspace that may be relevant to the current conversation",this.kind=o.PromptContextKind.Workspace,this.requi
redOnly=!0}async resolveContext(e,t,n,r){if(!n)return;const o=e.get(i.ChatMLFetcher),c=await o.fetchOne([{role:a.ChatRole.System,content:g},{role:a.ChatRo
le.User,content:n}],(async e=>{}),r,s.ChatUiKind.ConversationPanel,{temperature:.1,top_p:1,stop:["dontStopBelieving"]},{messageSource:"workspaceIntent"})