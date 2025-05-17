// lib/main.dart ───────────────────────────────────────────────────────────────
//  Flutter 3.19 · Material 3
//
//  * Sort / filter metric grid
//  * Cancel & retry for uploads + chat
//  * Two-pane layout on wide screens
//  * Dark-mode-safe colours
//  * Play video feature
//──────────────────────────────────────────────────────────────────────────────

// ignore_for_file: unused_element, unnecessary_null_comparison

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:multicast_dns/multicast_dns.dart';
import 'package:video_player/video_player.dart';

//──────────────────────────────────────────────────────────────────────────────
//  GLOBAL STATE
//──────────────────────────────────────────────────────────────────────────────

final apiBase =
    ValueNotifier<String>('http://mosquito-able-urgently.ngrok-free.app/');
final themeMode = ValueNotifier<ThemeMode>(ThemeMode.light);

//──────────────────────────────────────────────────────────────────────────────
//  DATA MODELS
//──────────────────────────────────────────────────────────────────────────────

class Metric {
  Metric.fromJson(Map j)
      : name = j['name'] ?? '',
        value = j['value'] ?? '',
        normalRange = j['normalRange'] ?? '',
        status = j['status'] ?? 'normal';

  final String name, value, normalRange, status;

  String get prettyValue {
    final num? n = num.tryParse(value.split(' ').first);
    if (n == null) return value;
    return n >= 100 ? n.toStringAsFixed(0) : n.toStringAsFixed(1);
  }

  Color color(BuildContext c) {
    final cs = Theme.of(c).colorScheme;
    return switch (status) {
      'warning' => cs.error,
      'danger' => cs.error,
      _ => cs.primary,
    };
  }
}

class PatientInfo {
  String sex = '';
  String weight = '';
  String height = '';
  String history = '';

  Map<String, String> toJson() => {
        'sex': sex,
        'weight': weight,
        'height': height,
        'history': history,
      };
}

//──────────────────────────────────────────────────────────────────────────────
//  ENTRYPOINT
//──────────────────────────────────────────────────────────────────────────────

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final sp = await SharedPreferences.getInstance();
  apiBase.value = sp.getString('api_base') ?? apiBase.value;
  themeMode.value =
      (sp.getBool('dark_mode') ?? false) ? ThemeMode.dark : ThemeMode.light;
  runApp(const _EchoApp());
}

class _EchoApp extends StatelessWidget {
  const _EchoApp({super.key});
  @override
  Widget build(BuildContext context) => ValueListenableBuilder<ThemeMode>(
        valueListenable: themeMode,
        builder: (_, mode, __) => MaterialApp(
          title: 'Patient Echo Dashboard',
          debugShowCheckedModeBanner: false,
          themeMode: mode,
          theme: _buildTheme(Brightness.light),
          darkTheme: _buildTheme(Brightness.dark),
          home: const HomePage(),
        ),
      );

  ThemeData _buildTheme(Brightness b) {
    final base = ThemeData(
      colorSchemeSeed: const Color(0xFF0081A7),
      useMaterial3: true,
      brightness: b,
    );
    return base.copyWith(
      textTheme: GoogleFonts.interTextTheme(base.textTheme),
    );
  }
}

//──────────────────────────────────────────────────────────────────────────────
//  HOME PAGE
//──────────────────────────────────────────────────────────────────────────────

enum SortMode { alpha, status }

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // upload / chat state
  PlatformFile? _video;
  bool _uploadBusy = false, _chatBusy = false, _uploadFailed = false;
  http.StreamedResponse? _ongoingUpload;

  // data
  final List<Metric> _metrics = [];
  final List<_ChatBubble> _chat = [];
  final PatientInfo _patient = PatientInfo();

  // sort / filter
  SortMode _sort = SortMode.alpha;
  bool _abnormalOnly = false;

  //──────────────── Video Picking ─────────────────

  Future<void> _pickVideo() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: kIsWeb,
    );
    if (res != null) setState(() => _video = res.files.single);
  }

  //──────────────── Metric Explanation ─────────────────

  Future<String> _fetchMetricExplanation(Metric m) async {
    final body = {
      'metric_name': m.name,
      'value': m.value,
      'normalRange': m.normalRange,
      'all_metrics': _metrics
          .map((e) => {
                'name': e.name,
                'value': e.value,
                'normalRange': e.normalRange,
                'status': e.status,
              })
          .toList(),
      'patient_info': _patient.toJson(),
    };
    final resp = await http.post(
      Uri.parse('${apiBase.value}/explain_metric'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (resp.statusCode == 200) {
      return jsonDecode(resp.body)['explanation'] as String;
    } else {
      throw Exception('Error ${resp.statusCode}: ${resp.body}');
    }
  }

  //──────────────── Upload + Metric Fetch ─────────────────

  Future<void> _getMetrics() async {
    if (_video == null) return _showSnack('Choose a video first');
    setState(() {
      _uploadBusy = true;
      _uploadFailed = false;
    });
    try {
      final req = http.MultipartRequest(
        'POST',
        Uri.parse('${apiBase.value}/analyse_video'),
      );
      http.MultipartFile part;
      if (_video!.bytes != null) {
        part = http.MultipartFile.fromBytes(
          'file',
          _video!.bytes!,
          filename: _video!.name,
          contentType: MediaType('video', 'mp4'),
        );
      } else {
        part = await http.MultipartFile.fromPath(
          'file',
          _video!.path!,
          filename: _video!.name,
          contentType: MediaType('video', 'mp4'),
        );
      }
      req.files.add(part);
      _ongoingUpload = await req.send();
      final body = await _ongoingUpload!.stream.bytesToString();
      if (_ongoingUpload!.statusCode != 200) {
        throw Exception('HTTP ${_ongoingUpload!.statusCode}: $body');
      }
      _metrics
        ..clear()
        ..addAll((jsonDecode(body)['metrics'] as List)
            .map((e) => Metric.fromJson(e)));
      _showSnack('Analysis complete ✔');
    } catch (e, st) {
      debugPrintStack(stackTrace: st, label: e.toString());
      _uploadFailed = true;
      _showSnack('Upload error: $e');
    } finally {
      _ongoingUpload = null;
      setState(() => _uploadBusy = false);
    }
  }

  void _cancelUpload() {
    _ongoingUpload?.stream.listen(null).cancel();
    setState(() => _uploadBusy = false);
  }

  //──────────────── Chat / Ask AI ─────────────────

  Future<void> _ask(String q, {bool think = false}) async {
    if (q.trim().isEmpty || _chatBusy) return;
    setState(() => _chat.add(_ChatBubble.user(q)));

    final systemData = _metrics
        .map((m) => '${m.name}: ${m.value} (Normal: ${m.normalRange})')
        .join('\n');

    final body = {
      'conversation_history': _chat.map((b) => b.toMap()).toList(),
      'query': q,
      'system_data': systemData,
      'patient_info': _patient.toJson(),
      'think': think,
    };

    final placeholder = _ChatBubble.assistant();
    setState(() {
      _chatBusy = true;
      _chat.add(placeholder);
    });

    final uri = Uri.parse('${apiBase.value}/ask');
    final req = http.post(uri,
        headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));

    try {
      final resp = await req;
      if (!mounted) return;
      final txt = resp.statusCode == 200
          ? (jsonDecode(resp.body)['response'] ?? '')
          : 'Error ${resp.statusCode}';
      setState(() => placeholder.setText(txt));
    } catch (e) {
      if (mounted) setState(() => placeholder.setText('❌ $e'));
    } finally {
      if (mounted) setState(() => _chatBusy = false);
    }
  }

  void _cancelChat() {
    setState(() => _chatBusy = false);
  }

  //──────────────── Helpers ─────────────────

  void _showSnack(String m) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m)));

  Future<bool> _ping(String base) async {
    try {
      final r = await http.get(Uri.parse('$base/ping')).timeout(
            const Duration(seconds: 2),
          );
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<String>> _discoverServers({
    Duration timeout = const Duration(seconds: 3),
  }) async {
    final client = MDnsClient(
      rawDatagramSocketFactory: (dynamic host, int _unused,
          {bool? reuseAddress, bool? reusePort, int? ttl}) {
        return RawDatagramSocket.bind(
          host,
          0,
          reuseAddress: true,
          reusePort: false,
          ttl: ttl ?? 1,
        );
      },
    );
    await client.start();
    final servers = <String>{};
    await for (final ptr in client
        .lookup<PtrResourceRecord>(
            ResourceRecordQuery.serverPointer('_echoserver._tcp.local'))
        .timeout(timeout, onTimeout: (_) => null)) {
      if (ptr == null) break;
      await for (final srv in client.lookup<SrvResourceRecord>(
          ResourceRecordQuery.service(ptr.domainName))) {
        await for (final ip in client.lookup<IPAddressResourceRecord>(
            ResourceRecordQuery.addressIPv4(srv.target))) {
          servers.add('http://${ip.address.address}:${srv.port}');
        }
      }
    }
    client.stop();
    return servers.toList();
  }

  //──────────────── Build UI ─────────────────

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;
    final wide = w >= 900;

    Widget metricsSection = Column(
      children: [
        _buildControlsBar(),
        const SizedBox(height: 12),
        if (_uploadBusy) _buildBusyCard(),
        if (_uploadFailed)
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: Text('Upload failed – tap retry.',
                style: TextStyle(
                    color: Theme.of(context).colorScheme.error,
                    fontWeight: FontWeight.w600)),
          ),
        _buildMetricGrid(),
      ],
    );

    Widget chatSection = _buildChatCard();

    return Scaffold(
      appBar: AppBar(
        elevation: 1,
        titleSpacing: 24,
        title: Text('Echo Dashboard',
            style:
                GoogleFonts.inter(fontWeight: FontWeight.w600, fontSize: 22)),
        actions: [
          IconButton(
            icon: Icon(themeMode.value == ThemeMode.dark
                ? Icons.light_mode_outlined
                : Icons.dark_mode_outlined),
            tooltip:
                themeMode.value == ThemeMode.dark ? 'Light mode' : 'Dark mode',
            onPressed: _toggleTheme,
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'API server',
            onPressed: _showServerDialog,
          ),
        ],
      ),
      body: SafeArea(
        child: Align(
          alignment: Alignment.topCenter,
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1100),
            child: wide
                ? Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Flexible(
                        flex: 3,
                        child: ListView(
                          padding: const EdgeInsets.all(16),
                          children: [
                            _buildPatientInfoCard(),
                            const SizedBox(height: 16),
                            _buildUploaderCard(), // now includes Annotate
                            const SizedBox(height: 24),
                            metricsSection,
                          ],
                        ),
                      ),
                      const VerticalDivider(width: 1),
                      Flexible(
                        flex: 2,
                        child: ListView(
                          padding: const EdgeInsets.all(16),
                          children: [chatSection],
                        ),
                      ),
                    ],
                  )
                : ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      _buildPatientInfoCard(),
                      const SizedBox(height: 16),
                      _buildUploaderCard(),
                      const SizedBox(height: 24),
                      metricsSection,
                      const SizedBox(height: 28),
                      chatSection,
                    ],
                  ),
          ),
        ),
      ),
    );
  }

  //──────────────── Sort / Filter Bar ─────────────────

  Widget _buildControlsBar() => Row(
        children: [
          DropdownButton<SortMode>(
            value: _sort,
            underline: const SizedBox(),
            onChanged: (v) => setState(() => _sort = v!),
            items: const [
              DropdownMenuItem(
                  value: SortMode.alpha, child: Text('A-Z (name)')),
              DropdownMenuItem(
                  value: SortMode.status, child: Text('Status (warn first)')),
            ],
          ),
          const SizedBox(width: 16),
          FilterChip(
            label: const Text('Abnormal only'),
            selected: _abnormalOnly,
            onSelected: (v) => setState(() => _abnormalOnly = v),
          ),
          const Spacer(),
          if (_uploadBusy)
            TextButton.icon(
              icon: const Icon(Icons.cancel_outlined),
              label: const Text('Cancel upload'),
              onPressed: _cancelUpload,
            ),
          if (_uploadFailed)
            TextButton.icon(
              icon: const Icon(Icons.replay_outlined),
              label: const Text('Retry'),
              onPressed: _getMetrics,
            ),
          if (_chatBusy)
            TextButton.icon(
              icon: const Icon(Icons.cancel_outlined),
              label: const Text('Cancel chat'),
              onPressed: _cancelChat,
            ),
        ],
      );

  //──────────────── Metric Grid ─────────────────

  List<Metric> get _sortedFilteredMetrics {
    var list = List<Metric>.from(_metrics);
    if (_abnormalOnly) {
      list = list.where((m) => m.status != 'normal').toList();
    }
    switch (_sort) {
      case SortMode.alpha:
        list.sort((a, b) => a.name.compareTo(b.name));
      case SortMode.status:
        int score(Metric m) =>
            m.status == 'normal' ? 2 : (m.status == 'warning' ? 0 : 1);
        list.sort((a, b) => score(a).compareTo(score(b)));
    }
    return list;
  }

  Widget _buildMetricGrid() => GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        shrinkWrap: true,
        padding: EdgeInsets.zero,
        gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
          maxCrossAxisExtent: 380,
          mainAxisSpacing: 16,
          crossAxisSpacing: 16,
          childAspectRatio: 2.4,
        ),
        itemCount: _sortedFilteredMetrics.length,
        itemBuilder: (_, i) => GestureDetector(
          onTap: () => _onMetricTap(_sortedFilteredMetrics[i]),
          child: _MetricTile(_sortedFilteredMetrics[i]),
        ),
      );

  //──────────────── Patient Info Card ─────────────────

  Widget _buildPatientInfoCard() => Card(
        elevation: 0.5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Patient Info',
                  style: GoogleFonts.inter(
                      fontSize: 18, fontWeight: FontWeight.w600)),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _patient.sex.isEmpty ? null : _patient.sex,
                    items: ['Male', 'Female', 'Other']
                        .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                        .toList(),
                    hint: const Text('Sex'),
                    onChanged: (v) => setState(() => _patient.sex = v!),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(labelText: 'Weight (kg)'),
                    keyboardType: TextInputType.number,
                    onChanged: (v) => _patient.weight = v,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(labelText: 'Height (cm)'),
                    keyboardType: TextInputType.number,
                    onChanged: (v) => _patient.height = v,
                  ),
                ),
              ]),
              const SizedBox(height: 12),
              TextField(
                decoration: const InputDecoration(
                  labelText: 'Medical history',
                  alignLabelWithHint: true,
                ),
                maxLines: 3,
                onChanged: (v) => _patient.history = v,
              ),
            ],
          ),
        ),
      );

  //──────────────── Uploader Card w/ Annotate ─────────────────
  Widget _buildUploaderCard() => Card(
        elevation: 0.5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Upload new study', /* … */
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      icon: const Icon(Icons.video_file_outlined),
                      label: Text(
                          _video == null ? 'Choose video' : _video!.name,
                          overflow: TextOverflow.ellipsis),
                      onPressed: _uploadBusy ? null : _pickVideo,
                    ),
                  ),
                  const SizedBox(width: 12),
                  FilledButton.icon(
                    icon: const Icon(Icons.cloud_upload_outlined),
                    label: const Text('Analyse'),
                    onPressed:
                        _uploadBusy || _video == null ? null : _getMetrics,
                  ),
                ],
              ),

              // ← NEW: always show Annotate below the row
              if (_video != null && !_uploadBusy) ...[
                const SizedBox(height: 12),
                // inside _buildUploaderCard(), replace the FilledButton block:

                FilledButton.icon(
                  icon: const Icon(Icons.play_circle_outline),
                  label: const Text('Play Video'),
                  onPressed: () {
                    final file =
                        _video?.path != null ? File(_video!.path!) : null;
                    if (file != null) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (_) => PlayVideoPage(videoFile: file)),
                      );
                    }
                  },
                ),
              ],
            ],
          ),
        ),
      );

  //──────────────── Busy Card ─────────────────

  Widget _buildBusyCard() => Card(
        margin: const EdgeInsets.only(top: 24),
        elevation: 0.5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(children: [
            Text('Analysing video…',
                style: GoogleFonts.inter(
                    fontSize: 16, fontWeight: FontWeight.w500)),
            const SizedBox(height: 20),
            SpinKitPulse(
              color: Theme.of(context).colorScheme.primary,
              size: 60,
            ),
          ]),
        ),
      );

  //──────────────── Metric Tap ─────────────────

  void _onMetricTap(Metric m) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(m.name),
        content: FutureBuilder<String>(
          future: _fetchMetricExplanation(m),
          builder: (ctx, snap) {
            if (snap.connectionState != ConnectionState.done) {
              return const SizedBox(
                height: 120,
                child: Center(child: CircularProgressIndicator()),
              );
            }
            if (snap.hasError) {
              return Text('Error: ${snap.error}');
            }
            return SingleChildScrollView(
              child: MarkdownBody(data: snap.data!),
            );
          },
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('Close')),
        ],
      ),
    );
  }

  //──────────────── Chat Card ─────────────────

  Widget _buildChatCard() => Card(
        elevation: 0.5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Text('Consult AI Cardiologist',
                  style: GoogleFonts.inter(
                      fontSize: 18, fontWeight: FontWeight.w600)),
              const SizedBox(height: 16),
              SizedBox(
                height: 340,
                child: ListView(
                  children: _chat.map((b) => b.build(context)).toList(),
                ),
              ),
              const Divider(height: 32),
              _InputBar(
                onSend: (q) => _ask(q, think: false),
                onThink: (q) => _ask(q, think: true),
                busy: _chatBusy,
              ),
            ],
          ),
        ),
      );

  //──────────────── Settings helpers ─────────────────

  void _toggleTheme() async {
    final newMode =
        themeMode.value == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    themeMode.value = newMode;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool('dark_mode', newMode == ThemeMode.dark);
    setState(() {});
  }

  Future<void> _showServerDialog() async {
    final ctrl = TextEditingController(text: apiBase.value);
    List<String> discovered = const [];
    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (ctx, setStateSB) => AlertDialog(
          title: const Text('API Server'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: ctrl,
                decoration:
                    const InputDecoration(hintText: 'http://192.168.x.x:8080'),
              ),
              const SizedBox(height: 16),
              Row(children: [
                const Text('LAN scan',
                    style: TextStyle(fontWeight: FontWeight.w500)),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.search),
                  tooltip: 'Scan for Echo servers',
                  onPressed: () async {
                    setStateSB(() => discovered = ['(scanning…)']);
                    final r = await _discoverServers();
                    setStateSB(() => r.isEmpty ? ['(none found)'] : r);
                  },
                ),
              ]),
              for (final url in discovered)
                ListTile(
                  dense: true,
                  enabled: url.startsWith('http'),
                  title: Text(url),
                  onTap: url.startsWith('http') ? () => ctrl.text = url : null,
                ),
            ],
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                final newUrl = ctrl.text.trim().replaceAll(RegExp(r'/+\$'), '');
                if (newUrl.isEmpty) return;
                _showSnack('Checking server…');
                if (await _ping(newUrl)) {
                  apiBase.value = newUrl;
                  final sp = await SharedPreferences.getInstance();
                  await sp.setString('api_base', newUrl);
                  if (mounted) setState(() {});
                  _showSnack('Server saved ✔');
                  Navigator.pop(ctx);
                } else {
                  _showSnack('Server unreachable');
                }
              },
              child: const Text('Save'),
            ),
          ],
        ),
      ),
    );
  }
}

//──────────────────────────────────────────────────────────────────────────────
//  METRIC TILE ─ overflow-safe & dark-mode aware
//──────────────────────────────────────────────────────────────────────────────

class _MetricTile extends StatelessWidget {
  const _MetricTile(this.m, {super.key});
  final Metric m;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final col = m.color(context);
    return Material(
      color: cs.surface,
      elevation: 1,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: col.withOpacity(.15),
              child: Icon(Icons.stacked_bar_chart_rounded, color: col),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(m.name,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.inter(
                          height: 1.1,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: cs.onSurface)),
                  const SizedBox(height: 2),
                  FittedBox(
                    fit: BoxFit.scaleDown,
                    alignment: Alignment.centerLeft,
                    child: Text(m.prettyValue,
                        style: GoogleFonts.inter(
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                            color: col)),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Tooltip(
              message: 'Normal: ${m.normalRange}',
              child: Text(m.normalRange,
                  style: GoogleFonts.inter(
                      fontSize: 12, color: cs.onSurfaceVariant)),
            ),
          ],
        ),
      ),
    );
  }
}

//──────────────────────────────────────────────────────────────────────────────
//  CHAT WIDGETS
//──────────────────────────────────────────────────────────────────────────────

class _ChatBubble {
  _ChatBubble.user(this.text)
      : isUser = true,
        _animate = false;
  _ChatBubble.assistant()
      : isUser = false,
        text = '',
        _animate = true;
  bool isUser;
  String text;
  bool _animate;

  Map<String, String> toMap() =>
      {'role': isUser ? 'user' : 'assistant', 'content': text};

  void setText(String newText) {
    text = newText;
    _animate = false;
  }

  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final align = isUser ? Alignment.centerRight : Alignment.centerLeft;
    final bubbleColor =
        isUser ? cs.primaryContainer : cs.surfaceVariant.withOpacity(0.9);
    final child = _animate || text.isEmpty
        ? (text.isEmpty
            ? const Text('•••')
            : AnimatedTextKit(
                animatedTexts: [
                  TyperAnimatedText(text,
                      speed: const Duration(milliseconds: 20),
                      textStyle: GoogleFonts.inter()),
                ],
                isRepeatingAnimation: false,
                totalRepeatCount: 1,
                displayFullTextOnTap: true,
              ))
        : MarkdownBody(data: text);
    return Align(
      alignment: align,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: bubbleColor,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(12),
            topRight: const Radius.circular(12),
            bottomLeft: Radius.circular(isUser ? 12 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 12),
          ),
        ),
        child: child,
      ),
    );
  }
}

class _InputBar extends StatefulWidget {
  const _InputBar({
    required this.onSend,
    required this.onThink,
    this.busy = false,
  });
  final ValueChanged<String> onSend;
  final ValueChanged<String> onThink;
  final bool busy;
  @override
  State<_InputBar> createState() => _InputBarState();
}

class _InputBarState extends State<_InputBar> {
  final _ctrl = TextEditingController();
  @override
  Widget build(BuildContext context) => Row(
        children: [
          Expanded(
            child: TextField(
              controller: _ctrl,
              enabled: !widget.busy,
              decoration: const InputDecoration(hintText: 'Ask a question…'),
              onSubmitted: _send,
            ),
          ),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: widget.busy ? null : () => _send(_ctrl.text),
            child: widget.busy
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.send),
          ),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: widget.busy ? null : () => _think(_ctrl.text),
            child: const Text('Think'),
          ),
        ],
      );

  void _send(String v) {
    if (v.trim().isEmpty) return;
    widget.onSend(v.trim());
    _ctrl.clear();
  }

  void _think(String v) {
    if (v.trim().isEmpty) return;
    widget.onThink(v.trim());
    _ctrl.clear();
  }
}

//──────────────────────────────────────────────────────────────────────────────
//  PLAY-VIDEO SCREEN  (simple bookmarks, no ROI)
//──────────────────────────────────────────────────────────────────────────────
class PlayVideoPage extends StatefulWidget {
  final File videoFile;
  const PlayVideoPage({required this.videoFile, super.key});

  @override
  State<PlayVideoPage> createState() => _PlayVideoPageState();
}

class _PlayVideoPageState extends State<PlayVideoPage> {
  late final VideoPlayerController _ctl;
  final List<Duration> _bookmarks = [];

  @override
  void initState() {
    super.initState();
    _ctl = VideoPlayerController.file(widget.videoFile)
      ..initialize().then((_) => setState(() {}));
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  String _fmt(Duration d) =>
      '${d.inMinutes.remainder(60).toString().padLeft(2, '0')}:'
      '${d.inSeconds.remainder(60).toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    if (!_ctl.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      appBar: AppBar(title: const Text('Play Video')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: _ctl.value.aspectRatio,
                  child: VideoPlayer(_ctl),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: VideoProgressIndicator(_ctl, allowScrubbing: true),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.bookmark_add_outlined),
                    tooltip: 'Save moment',
                    onPressed: () =>
                        setState(() => _bookmarks.add(_ctl.value.position)),
                  ),
                  Expanded(
                    child: Wrap(
                      spacing: 8,
                      children: _bookmarks
                          .map(
                            (bm) => ActionChip(
                              label: Text(_fmt(bm)),
                              onPressed: () {
                                _ctl.seekTo(bm);
                              },
                            ),
                          )
                          .toList(),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // trigger play/pause (returns a Future, but we don’t await it here)
          if (_ctl.value.isPlaying) {
            _ctl.pause();
          } else {
            _ctl.play();
          }
          // then rebuild immediately
          setState(() {});
        },
        child: Icon(_ctl.value.isPlaying ? Icons.pause : Icons.play_arrow),
      ),
    );
  }
}
