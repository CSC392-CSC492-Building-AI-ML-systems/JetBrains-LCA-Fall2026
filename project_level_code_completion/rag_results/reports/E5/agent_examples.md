# What the agent did to RAG F's lines: random examples (N=1,753, CodeLlama-7B)

Each kind below is a random draw (seed 0) from all lines of that kind; the number in brackets is how many lines there are. 'Needed names' are the names the target line uses that the repository defines in another file (the lexical screen of the failure split); 'missing' = not present in the prompt the model saw. A name being present does not mean its definition was useful.


## Broke: RAG F right, agent wrong, the needed names were visible in both prompts [63 lines]

### seriaati__shenhe_bot, `cmd/flow_find.py`, line 59 (infile; datapoint 66)

Code before the line (last 6 lines of the prefix):
```python
                        author = self.bot.get_user(find['authorID'])
                        acceptUser = self.bot.get_user(user['discordID'])
                        if find['type']==1:
                            await author.send(f"[成功接受委託] {acceptUser.mention} 接受了你的 {find['title']} 委託")
                            await acceptUser.send(f"[成功接受委託] 你接受了 {author.mention} 的 {find['title']} 委託")
                            await channel.send(f"✅ {acceptUser.mention} 已接受 {author.mention} 的 {find['title...
```
```text
truth      : elif find['type']==2:
RAG F      : elif find['type']==2:   [right]
agent      : await channel.send(f"{acceptUser.mention} 請在 10 分鐘內回復委託")   [wrong]
```
- needed names from other files: ['find']; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "FlowCog" -> submit_context add=[chunk #76, chunk #110]
- added to the context: cmd/flow.py 25-25, cmd/flow.py 673-674; displaced from RAG F's context: cmd/flow.py 130-169, asset/global_vars.py 21-23

### mampfes__hacs_waste_collection_schedule, `custom_components/waste_collection_schedule/waste_collection_schedule/source/aw_harburg.py`, line 72 (common; datapoint 155)

Code before the line (last 6 lines of the prefix):
```python
        soup = BeautifulSoup(str(select_content), features="html.parser")
        options_content = soup.find_all("option")
        level_1_ids = {}
        for option in options_content:
            # Ignore the "Bitte wählen..."
            if option.get("value")!="0":
```
```text
truth      : level_1_ids[option.text] = option.get("value")
RAG F      : level_1_ids[option.text] = option.get("value")   [right]
agent      : level_1_ids[option.get("value")] = option.get_text()   [wrong]
```
- needed names from other files: ['value']; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "Collection, ICS, BeautifulSoup, requests" -> submit_context add=[chunk #236, chunk #228, chunk #329, chunk #224]
- added to the context: custom_components/waste_collection_schedule/waste_collection_schedule/source/wuerzburg_de.py 0-93, custom_components/waste_collection_schedule/waste_collection_schedule/source/landkreis_wittmund_de.py 0-120; displaced from RAG F's context: custom_components/waste_collection_schedule/waste_collection_schedule/source/wuerzburg_de.py 23-58, custom_components/waste_collection_schedule/waste_collection_schedule/source/landkreis_wittmund_de.py 37-62

### qiboteam__qibocal, `src/qcvv/fitting/methods.py`, line 254 (inproject; datapoint 208)

Code before the line (last 6 lines of the prefix):
```python
        ]

    try:
        popt, pcov = curve_fit(
            exp, time.values, voltages.values, p0=pguess, maxfev=2000000
        )
```
```text
truth      : t1 = abs(1 / popt[2])
RAG F      : t1 = abs(1 / popt[2])   [right]
agent      : t1 = 1.0 / popt[2]   [wrong]
```
- needed names from other files: ['t1']; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "curve_fit" -> submit_context add=[chunk #66]
- added to the context: src/qibocal/fitting/methods.py 0-360; displaced from RAG F's context: src/qibocal/fitting/methods.py 222-268, src/qibocal/fitting/methods.py 107-146, src/qibocal/fitting/methods.py 291-314


## Broke: RAG F right, agent wrong, the line needs nothing from other files [53 lines]

### single-cell-data__tiledb-soma, `apis/python/src/tiledbsc/assay_matrix.py`, line 195 (random; datapoint 17)

Code before the line (last 6 lines of the prefix):
```python
                # Convert the chunk to a COO matrix.
                chunk_coo = matrix[permutation[i:i2]].tocoo()

                s2 = util.get_start_stamp()

                # Write the chunk-COO to TileDB.
```
```text
truth      : d0 = sorted_row_names[chunk_coo.row + i]
RAG F      : d0 = sorted_row_names[chunk_coo.row + i]   [right]
agent      : A[sorted_row_names[i:i2], col_names[chunk_coo.col]] = chunk_coo.data   [wrong]
```
- needed names from other files: none; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "util.get_sort_and_permutation" -> submit_context add=[chunk #113, chunk #112]
- added to the context: apis/python/src/tiledbsc/util.py 50-68, apis/python/src/tiledbsc/util.py 28-47; displaced from RAG F's context: apis/python/src/tiledbsc/soma.py 742-781, apis/python/src/tiledbsc/soma.py 722-761

### seldomqa__seldom, `seldom/testdata/random_func.py`, line 196 (non_informative; datapoint 94)

Code before the line (last 6 lines of the prefix):
```python
    Decide if we should perform this action, this is just a simple way to do something
    I do in tests every now and again

    :Example:
        # EXAMPLE -- simple yes or no question
        if testdata.yes():
```
```text
truth      : # do this
RAG F      : # do this   [right]
agent      : # do something   [wrong]
```
- needed names from other files: none; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "zh_names_male, zh_names_female, zh_last_name, words_str" -> submit_context add=[chunk #362]
- added to the context: seldom/testdata/__init__.py 0-382; displaced from RAG F's context: seldom/testdata/__init__.py 181-228, seldom/testdata/__init__.py 168-178, seldom/testdata/__init__.py 130-136, seldom/testdata/__init__.py 231-246

### datafold__data-diff, `data_diff/databases/base.py`, line 188 (infile; datapoint 103)

Code before the line (last 6 lines of the prefix):
```python
        elif len(path) != 2:
            raise ValueError(f"{self.name}: Bad table path for {self}: '{'.'.join(path)}'. Expected form: s...

        return path

    def parse_table_name(self, name: str) -> DbPath:
```
```text
truth      : return parse_table_name(name)
RAG F      : return parse_table_name(name)   [right]
agent      : return tuple(name.split("."))   [wrong]
```
- needed names from other files: none; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "DbPath" -> submit_context add=[chunk #55, chunk #63, chunk #0, chunk #50, chunk #16]
- added to the context: data_diff/sql.py 42-45, data_diff/sql.py 101-105; displaced from RAG F's context: data_diff/diff_tables.py 0-429, tests/test_normalize_fields.py 93-115


## Broke: RAG F right, agent wrong, a needed name disappeared from the prompt (eviction) [20 lines]

### seriaati__shenhe_bot, `cmd/flow_find.py`, line 73 (inproject; datapoint 66)

Code before the line (last 6 lines of the prefix):
```python
                        elif find['type']==4:
                            await author.send(f"✅ {acceptUser.mention} 接受了你的 {find['title']} 幫助")
                            await acceptUser.send(f"✅ 你接受了 {author.mention} 的 {find['title']} 幫助")
                            await channel.send(f"✅ {acceptUser.mention} 接受 {author.mention} 的 {find['title'...
                        if find['type'] == 4:
                            embedDM = global_vars.defaultEmbed("結算單",f"當對方完成幫忙的內容時, 請按 🆗來結算flow幣\n按下後, 你的fl...
```
```text
truth      : global_vars.setFooter(embedDM)
RAG F      : global_vars.setFooter(embedDM)   [right]
agent      : embedDM.set_footer(text=f"{find['title']} 幫忙結算單")   [wrong]
```
- needed names from other files: ['setFooter']; missing in RAG F's prompt: none; missing in the agent's prompt: ['setFooter']
- planner: search_code "FlowCog" -> submit_context add=[chunk #76, chunk #110, chunk #77, chunk #109, chunk #100]
- added to the context: cmd/flow.py 25-25, cmd/flow.py 673-674; displaced from RAG F's context: cmd/flow.py 130-169, asset/global_vars.py 21-23

### seldomqa__seldom, `seldom/testdata/random_func.py`, line 253 (inproject; datapoint 94)

Code before the line (last 6 lines of the prefix):
```python
        count = random.randint(1, 20)

    if not words:
        words = words_str

    ret_words = random.sample(words, count)
```
```text
truth      : return ret_words if not as_str else ' '.join(ret_words)
RAG F      : return ret_words if not as_str else ' '.join(ret_words)   [right]
agent      : if as_str:   [wrong]
```
- needed names from other files: ['join']; missing in RAG F's prompt: none; missing in the agent's prompt: ['join']
- planner: search_code "hashlib" -> submit_context add=[chunk #362]
- added to the context: seldom/testdata/__init__.py 0-382; displaced from RAG F's context: seldom/testdata/__init__.py 231-246, seldom/testdata/__init__.py 168-178, seldom/testdata/__init__.py 253-275, seldom/testdata/__init__.py 106-120

### pnxenopoulos__awpy, `awpy/parser/demoparser.py`, line 608 (infile; datapoint 100)

Code before the line (last 6 lines of the prefix):
```python
    def clean_rounds(
        self,
        remove_no_frames=True,
        remove_warmups=True,
        remove_knifes=True,
        remove_bad_timings=True,
```
```text
truth      : remove_excess_players=True,
RAG F      : remove_excess_players=True,   [right]
agent      : remove_bad_timings_threshold=10,   [wrong]
```
- needed names from other files: ['remove_excess_players']; missing in RAG F's prompt: none; missing in the agent's prompt: ['remove_excess_players']
- planner: search_code "logger" -> find_definition "logger" -> search_code "import logging" -> submit_context add=[chunk #206]
- added to the context: csgo/parser/demoparser.py 0-922; displaced from RAG F's context: csgo/parser/demoparser.py 602-641, csgo/parser/demoparser.py 622-661, csgo/parser/demoparser.py 899-921


## Fixed: RAG F wrong, agent right [34 lines]

### aristanetworks__ansible-avd, `ansible_collections/arista/avd/plugins/module_utils/eos_designs_facts.py`, line 377 (commited; datapoint 109)

Code before the line (last 6 lines of the prefix):
```python
    @cached_property
    def mgmt_ip(self):
        return get(self._switch_data_combined, "mgmt_ip")

    @cached_property
    def platform(self):
```
```text
truth      : return get(self._switch_data_combined, "platform")
RAG F      : return get(self._switch_data_combined, "platform", required=True)   [wrong]
agent      : return get(self._switch_data_combined, "platform")   [right]
```
- needed names from other files: none; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "platform" -> find_definition "platform" -> search_code "platform" -> find_definition "platform" -> search_code "platform" -> submit_context add=[chunk #8876]
- added to the context: ansible_collections/arista/avd/roles/eos_cli_config_gen/templates/documentation/platform.j2 0-39; displaced from RAG F's context: ansible_collections/arista/avd/roles/eos_designs/doc/fabric-topology-v4.0.md 120-159, ansible_collections/arista/avd/molecule/evpn_underlay_ebgp_overlay_ebgp/inventory/group_vars/DC1_FABRIC.yml 260-280

### alan-turing-institute__autspaces, `server/apps/main/helpers.py`, line 191 (inproject; datapoint 131)

Code before the line (last 6 lines of the prefix):
```python


def make_uuid():
    return str(uuid.uuid1())

def delete_PE(uuid, ohmember):
```
```text
truth      : if PublicExperience.objects.filter(experience_id=uuid, open_humans_member=ohmember).exists():
RAG F      : """Deletes a given file id and uuid from local PublicExperiences database.   [wrong]
agent      : if PublicExperience.objects.filter(experience_id=uuid, open_humans_member=ohmember).exists():   [right]
```
- needed names from other files: ['PublicExperience']; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "delete_PE" -> submit_context add=[chunk #56]
- added to the context: server/apps/main/views.py 157-159; displaced from RAG F's context: server/apps/main/views.py 247-257, 00-project-documentation/platform-design/README.md 0-4

### bmaltais__kohya_ss, `finetune_gui.py`, line 623 (commited; datapoint 200)

Code before the line (last 6 lines of the prefix):
```python
                create_caption = gr.Checkbox(
                    label='Generate caption database', value=True
                )
                create_buckets = gr.Checkbox(
                    label='Generate image buckets', value=True
                )
```
```text
truth      : train = gr.Checkbox(label='Train model', value=True)
RAG F      : with gr.Row():   [wrong]
agent      : train = gr.Checkbox(label='Train model', value=True)   [right]
```
- needed names from other files: none; missing in RAG F's prompt: none; missing in the agent's prompt: none
- planner: search_code "os" -> search_code "shutil" -> submit_context add=[chunk #91]
- added to the context: library/dreambooth_folder_creation_gui.py 0-197; displaced from RAG F's context: library/basic_caption_gui.py 58-114


## RAG F retrieval miss that the agent filled (every needed name now visible), all of them [7 lines]

### Significant-Gravitas__Auto-GPT, `autogpt/routes/agent_protocol.py`, line 63 (inproject; datapoint 2)

Code before the line (last 6 lines of the prefix):
```python
                "additional_input": "python/code",
                "artifacts": [],
                "steps": []
            }
    """
    agent = request["agent"]
```
```text
truth      : task_request = await agent.create_task(task_request)
RAG F      : task = await agent.db.create_task(   [wrong]
agent      : task = await agent.db.create_task(   [wrong]
```
- needed names from other files: ['create_task']; missing in RAG F's prompt: ['create_task']; missing in the agent's prompt: none
- planner: search_code "Agent" -> search_code "Agent" -> search_code "Agent" -> search_code "Agent" -> search_code "Agent" -> submit_context add=[chunk #140]
- added to the context: autogpt/agent_protocol/middlewares.py 20-41; displaced from RAG F's context: autogpt/agent_protocol/models.py 66-72, autogpt/agent.py 45-63

### LAION-AI__Open-Assistant, `backend/oasst_backend/api/v1/management.py`, line 24 (inproject; datapoint 59)

Code before the line (last 6 lines of the prefix):
```python
def _prepare_conversation(messages: list[Post]) -> protocol.Conversation:
    conv_messages = []
    for message in messages:
        if not isinstance(message.payload.payload, PostPayload):
            raise OasstError("Server error", OasstErrorCode.SERVER_ERROR, HTTPStatus.INTERNAL_SERVER_ERROR)
        conv_messages.append(
```
```text
truth      : protocol.ConversationMessage(text=message.payload.payload.text, is_assistant=(message.role == "assistant"))
RAG F      : protocol.Message(   [wrong]
agent      : protocol.ConversationMessage(   [wrong]
```
- needed names from other files: ['ConversationMessage']; missing in RAG F's prompt: ['ConversationMessage']; missing in the agent's prompt: none
- planner: search_code "protocol.Conversation" -> find_definition "protocol.Conversation" -> submit_context add=[chunk #325]
- added to the context: oasst-shared/oasst_shared/schemas/protocol.py 34-37; displaced from RAG F's context: backend/main.py 0-52, backend/oasst_backend/exceptions.py 59-61

### seriaati__shenhe_bot, `cmd/group.py`, line 483 (non_informative; datapoint 65)

Code before the line (last 6 lines of the prefix):
```python
                    global_vars.setFooter(embed)
                    await embedAsk.edit(embed=embed)
            if found == False:
                embed = global_vars.embedNoGroup
                global_vars.setFooter(embed)
                await embedAsk.edit(embed=embed)
```
```text
truth      : def setup(bot):
RAG F      : @bot.command()   [wrong]
agent      : def setup(bot):   [right]
```
- needed names from other files: ['bot', 'setup']; missing in RAG F's prompt: ['setup']; missing in the agent's prompt: ['setup']
- planner: search_code "groups" -> submit_context add=[chunk #45]
- added to the context: asset/global_vars.py 2-12; displaced from RAG F's context: run.py 451-459

### seriaati__shenhe_bot, `cogs/wish.py`, line 52 (inproject; datapoint 69)

Code before the line (last 6 lines of the prefix):
```python
        embed_list = []
        for l in split_list:
            embed_str = ''
            for w in l:
                embed_str += f'{w}\n'
            embed_list.append(defaultEmbed('詳細祈願紀錄', embed_str))
```
```text
truth      : await WishPaginator(i, embed_list).start(embeded=True)
RAG F      : paginator = WishPaginator(embed_list)   [wrong]
agent      : paginator = WishPaginator(i, embed_list)   [wrong]
```
- needed names from other files: ['WishPaginator', 'start']; missing in RAG F's prompt: ['start']; missing in the agent's prompt: none
- planner: search_code "WishCog" -> search_code "WishPaginator" -> search_code "WishPaginator" -> search_code "WishPaginator" -> search_code "WishPaginator" -> submit_context add=[chunk #258]
- added to the context: utility/WishPaginator.py 91-91; displaced from RAG F's context: cogs/gvaway.py 266-278

### latchbio__latch, `latch/services/register/register.py`, line 393 (inproject; datapoint 74)

Code before the line (last 6 lines of the prefix):
```python
            if ctx.pkg_root.joinpath("data") in file.parents:
                continue
            key = str(file.relative_to(ctx.pkg_root))
            commit_files[key] = open(file, "rb")

    commit_response = requests.post(
```
```text
truth      : url=ctx.latch_commit_api_url,
RAG F      : ctx.latch_commit_api_url,   [wrong]
agent      : ctx.latch_commit_api_url,   [wrong]
```
- needed names from other files: ['url']; missing in RAG F's prompt: ['url']; missing in the agent's prompt: none
- planner: search_code "requests" -> submit_context add=[chunk #166]
- added to the context: latch/services/login.py 68-90; displaced from RAG F's context: latch/services/register/models.py 67-87, tests/fixtures.py 0-25

### italia__spid-cie-oidc-django, `spid_cie_oidc/authority/tests/test_02_trust_anchor_intermediary.py`, line 35 (commited; datapoint 179)

Code before the line (last 6 lines of the prefix):
```python
class TrustChainTest(TestCase):
    def setUp(self):
        self.ta_conf = FederationEntityConfiguration.objects.create(**ta_conf_data)
        self.rp_conf = FederationEntityConfiguration.objects.create(**rp_conf)

        self.rp_profile = FederationEntityProfile.objects.create(**RP_PROFILE)
```
```text
truth      : self.rp = FederationDescendant.objects.create(**rp_onboarding_data)
RAG F      : self.rp_conf.federation_entity_profile.add(self.rp_profile)   [wrong]
agent      : self.rp_conf.profile.add(self.rp_profile)   [wrong]
```
- needed names from other files: ['FederationDescendant']; missing in RAG F's prompt: ['FederationDescendant']; missing in the agent's prompt: none
- planner: search_code "FederationEntityProfile" -> submit_context add=[chunk #138, chunk #152]
- added to the context: spid_cie_oidc/onboarding/admin.py 0-68, spid_cie_oidc/onboarding/models.py 92-93; displaced from RAG F's context: spid_cie_oidc/onboarding/models.py 0-39

### qiboteam__qibocal, `src/qibocal/plots/spectroscopies.py`, line 87 (inproject; datapoint 207)

Code before the line (last 6 lines of the prefix):
```python
    if len(data_fast) > 0 and len(data_fit) > 0:
        freqrange = np.linspace(
            min(data_fast.get_values("frequency", "GHz")),
            max(data_fast.get_values("frequency", "GHz")),
            2 * len(data_fast),
        )
```
```text
truth      : params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
RAG F      : fig.add_trace(   [wrong]
agent      : msr = lorenzian(freqrange, *data_fit.get_values("popt0", "popt1", "popt2", "popt3"))   [wrong]
```
- needed names from other files: ['df']; missing in RAG F's prompt: ['df']; missing in the agent's prompt: none
- planner: search_code "Data, DataUnits" -> search_code "random_data_units" -> submit_context add=[chunk #93]
- added to the context: src/qibocal/tests/test_data.py 9-25; displaced from RAG F's context: src/qibocal/fitting/methods.py 315-357


## RAG F retrieval miss that stays a miss under the agent [164 lines]

### intelligent-machine-learning__dlrover, `dlrover/python/master/master.py`, line 97 (inproject; datapoint 86)

Code before the line (last 6 lines of the prefix):
```python
            job_uuid, args.namespace, args.cluster, args.user
        )

    def prepare(self):
        # Composite the components
        if self.task_manager and self.node_manager:
```
```text
truth      : self.task_manager.set_task_timeout_callback(
RAG F      : self.task_manager.add_pod_event_callback(   [wrong]
agent      : self.task_manager.add_pod_event_callback(   [wrong]
```
- needed names from other files: ['set_task_timeout_callback']; missing in RAG F's prompt: ['set_task_timeout_callback']; missing in the agent's prompt: ['set_task_timeout_callback']
- planner: search_code "task_manager and node_manager" -> search_code "task_manager and node_manager" -> search_code "task_manager and node_manager" -> search_code "task_manager and node_manager" -> search_code "task_manager and node_manager" -> submit_context add=[]
- added to the context: nothing; displaced from RAG F's context: nothing

### princeton-vl__infinigen, `worldgen/surfaces/templates/fishbody.py`, line 1013 (inproject; datapoint 189)

Code before the line (last 6 lines of the prefix):
```python
    attributes = [
        'Color variations',
        'offset2'
    ]
        shader = shader_fish_body_gold
    else:
```
```text
truth      : surface.add_geomod(obj, geometry_fish_body, input_kwargs=geo_kwargs, attributes=attributes)
RAG F      : shader = shader_fish_body_regular   [wrong]
agent      : shader = shader_fish_body_regular   [wrong]
```
- needed names from other files: ['add_geomod', 'obj']; missing in RAG F's prompt: ['add_geomod']; missing in the agent's prompt: ['add_geomod']
- planner: search_code "shader_fish_body_gold" -> find_definition "shader_fish_body_gold" -> search_code "shader_eyeball_fish" -> search_code "shader_eyeball_fish" -> search_code "shader_eyeball_fish" -> submit_context add=[chunk #148]
- added to the context: nothing; displaced from RAG F's context: nothing

### mage-ai__mage-ai, `mage_ai/tests/data_cleaner/cleaning_rules/test_remove_collinear_columns.py`, line 8 (non_informative; datapoint 220)

Code before the line (last 6 lines of the prefix):
```python
from pandas.util.testing import assert_frame_equal
import numpy as np
import pandas as pd


class RemoveCollinearColumnsTests(TestCase):
```
```text
truth      : def setUp(self):
RAG F      : def test_remove_collinear_columns(self):   [wrong]
agent      : def test_remove_collinear_columns(self):   [wrong]
```
- needed names from other files: ['setUp']; missing in RAG F's prompt: ['setUp']; missing in the agent's prompt: ['setUp']
- planner: search_code "RemoveCollinearColumns" -> submit_context add=[chunk #56]
- added to the context: nothing; displaced from RAG F's context: nothing

